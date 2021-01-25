#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "FreeRTOS.h"
#include "mt3620.h"
#include "printf.h"
#include "task.h"
#include <semphr.h>

#include "os_hal_mbox.h"
#include "os_hal_mbox_shared_mem.h"
#include "os_hal_uart.h"

/******************************************************************************/
/* Configurations */
/******************************************************************************/
/* UART */
static const uint8_t uart_port_num = OS_HAL_UART_PORT0;

/* MailBox */
#define PAYLOAD_START 20
#define MAX_INTERCORE_BUF_SIZE 1024

/* FreeRTOS Task Stack */
#define APP_STACK_SIZE_BYTES 8192

/* Image */
#if defined PERSON_DETECTION_DEMO
#define IMAGE_WIDTH 96
#define IMAGE_HEIGHT 96
#define IMAGE_DEPTH 1
#define MAX_PIECE 9
#elif defined CIFAR10_DEMO
#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32
#define IMAGE_DEPTH 3
#define MAX_PIECE 3
#elif defined EMERGENCY_DETECT
#define SERIES_LENGTH 512
#define SERIES_FEATURE 6
#define MAX_PIECE 3
#endif

/******************************************************************************/
/* Global Variables */
/******************************************************************************/
#if defined PERSON_DETECTION_DEMO
static const char *label[] = {"unknown", "person", "no person"};
extern void person_detection_setup();
extern int person_detection_loop(uint8_t *input_buf);
#elif defined CIFAR10_DEMO
static const char *label[] = {"Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"};
extern void cifar10_setup();
extern int cifar10_invoke(uint8_t *input_buf);
#elif defined EMERGENCY_DETECT
static const char *label[] = {"UNKNOWN", "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"};
extern void emergency_detect_setup();
extern int emergency_detect_loop(uint8_t *input_buf);
#endif

#if (defined PERSON_DETECTION_DEMO) || (defined CIFAR10_DEMO)
static uint8_t ImgBuf[IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH];
#elif defined EMERGENCY_DETECT
static uint8_t InputBuf[SERIES_LENGTH * SERIES_FEATURE];
#endif

/* Mailbox semaphore */
SemaphoreHandle_t blockDeqSema;
SemaphoreHandle_t blockFifoSema;

/* Mailbox bitmap for IRQ enable. bit_0 and bit_1 are used to communicate with HL_APP */
static const uint32_t mbox_irq_status = 0x3;

/******************************************************************************/
/* Applicaiton Hooks */
/******************************************************************************/
/* Hook for "stack over flow". */
void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName) {
  printf("%s: %s\n", __func__, pcTaskName);
}

/* Hook for "memory allocation failed". */
void vApplicationMallocFailedHook(void) {
  printf("%s\n", __func__);
}

/* Hook for "printf". */
void _putchar(char character) {
  mtk_os_hal_uart_put_char(uart_port_num, character);
  if (character == '\n')
    mtk_os_hal_uart_put_char(uart_port_num, '\r');
}

/******************************************************************************/
/* Functions */
/******************************************************************************/
/* Mailbox Fifo Interrupt handler.
 * Mailbox Fifo Interrupt is triggered when mailbox fifo been R/W.
 *     data->event.channel: Channel_0 for A7, Channel_1 for the other M4.
 *     data->event.ne_sts: FIFO Non-Empty.interrupt
 *     data->event.nf_sts: FIFO Non-Full interrupt
 *     data->event.rd_int: Read FIFO interrupt
 *     data->event.wr_int: Write FIFO interrupt
*/
void mbox_fifo_cb(struct mtk_os_hal_mbox_cb_data *data) {
  BaseType_t higher_priority_task_woken = pdFALSE;

  if (data->event.channel == OS_HAL_MBOX_CH0) {
    /* A7 core write data to mailbox fifo. */
    if (data->event.wr_int) {
      xSemaphoreGiveFromISR(blockFifoSema,
                            &higher_priority_task_woken);
      portYIELD_FROM_ISR(higher_priority_task_woken);
    }
  }
}

/* SW Interrupt handler.
 * SW interrupt is triggered when:
 *    1. A7 read/write the shared memory.
 *    2. The other M4 triggers SW interrupt.
 *     data->swint.swint_channel: Channel_0 for A7, Channel_1 for the other M4.
 *     Channel_0:
 *         data->swint.swint_sts bit_0: A7 read data from mailbox
 *         data->swint.swint_sts bit_1: A7 write data to mailbox
 *     Channel_1:
 *         data->swint.swint_sts bit_0: M4 sw interrupt
*/
void mbox_swint_cb(struct mtk_os_hal_mbox_cb_data *data) {
  BaseType_t higher_priority_task_woken = pdFALSE;

  if (data->swint.channel == OS_HAL_MBOX_CH0) {
    if (data->swint.swint_sts & (1 << 1)) {
      xSemaphoreGiveFromISR(blockDeqSema,
                            &higher_priority_task_woken);
      portYIELD_FROM_ISR(higher_priority_task_woken);
    }
  }
}

static void Init_Mailbox() {
  struct mbox_fifo_event mask;

  /* Open the MBOX channel of A7 <-> M4 */
  mtk_os_hal_mbox_open_channel(OS_HAL_MBOX_CH0);

  blockDeqSema = xSemaphoreCreateBinary();
  blockFifoSema = xSemaphoreCreateBinary();

  /* Register interrupt callback */
  mask.channel = OS_HAL_MBOX_CH0;
  mask.ne_sts = 0; /* FIFO Non-Empty interrupt */
  mask.nf_sts = 0; /* FIFO Non-Full interrupt */
  mask.rd_int = 0; /* Read FIFO interrupt */
  mask.wr_int = 1; /* Write FIFO interrupt */
  mtk_os_hal_mbox_fifo_register_cb(OS_HAL_MBOX_CH0, mbox_fifo_cb, &mask);
  mtk_os_hal_mbox_sw_int_register_cb(OS_HAL_MBOX_CH0, mbox_swint_cb, mbox_irq_status);
}

static void NN_Task(void *pParameters) {
  uint32_t piece = 0;
  uint32_t recvSize = MAX_INTERCORE_BUF_SIZE + PAYLOAD_START;
  uint8_t recvBuf[MAX_INTERCORE_BUF_SIZE + PAYLOAD_START];
  uint8_t top_index;
  BufferHeader *outbound, *inbound;
  uint32_t sharedBufSize = 0;
  uint32_t time_start, exec_time;

  printf("NN Task Started\n");

  /* Initialize Mailbox */
  Init_Mailbox();

  /* Get mailbox buffers */
  if (GetIntercoreBuffers(&outbound, &inbound, (u32 *)&sharedBufSize) == -1) {
    printf("ERROR: GetIntercoreBuffers failed\r\n");
    while (1)
      ;
  }

  /* NN Init */
#if defined PERSON_DETECTION_DEMO
  person_detection_setup();
#elif defined CIFAR10_DEMO
  cifar10_setup();
#elif defined EMERGENCY_DETECT
  emergency_detect_setup();
#endif

  while (1) {
    /* waiting for incoming data */
    if (DequeueData(outbound, inbound, sharedBufSize, &recvBuf[0], (u32 *)&recvSize) == -1) {
      continue;
    }

    /* 3072 = 1024 x 3, as the maximum allowed user payload is 1024, we need split into 3 buffer. */
    /* (HL and RT core must be synced) */
    if (piece < MAX_PIECE) {

// memcpy(&ImgBuf[piece * MAX_INTERCORE_BUF_SIZE], &recvBuf[PAYLOAD_START], MAX_INTERCORE_BUF_SIZE);
#if (defined PERSON_DETECTION_DEMO) || (defined CIFAR10_DEMO)
      memcpy(&ImgBuf[piece * MAX_INTERCORE_BUF_SIZE], &recvBuf[PAYLOAD_START], MAX_INTERCORE_BUF_SIZE);
#elif defined EMERGENCY_DETECT
      memcpy(&InputBuf[piece * MAX_INTERCORE_BUF_SIZE], &recvBuf[PAYLOAD_START], MAX_INTERCORE_BUF_SIZE);
#endif
      piece++;
    }

    if (piece == MAX_PIECE) {
      piece = 0;
      time_start = xTaskGetTickCount();

#if defined PERSON_DETECTION_DEMO
      top_index = person_detection_loop((uint8_t *)&ImgBuf[0]);
#elif defined CIFAR10_DEMO
      top_index = cifar10_invoke((uint8_t *)&ImgBuf[0]);
#elif defined EMERGENCY_DETECT
      top_index = emergency_detect_loop((uint8_t *)&InputBuf[0]);
#endif

      printf("%s\r\n", label[top_index]);

      exec_time = xTaskGetTickCount() - time_start;
      printf("exec_time = %ld\r\n\r\n", exec_time);

      // Send the result back to HL core
      recvBuf[PAYLOAD_START] = top_index;
      for (int k = 0; k < 4; k++) {
        recvBuf[PAYLOAD_START + 1 + k] = exec_time & 0xFF;
        exec_time = exec_time >> 8;
      }
      EnqueueData(inbound, outbound, sharedBufSize, &recvBuf[0], PAYLOAD_START + 5);
    }
  }
}

_Noreturn void RTCoreMain(void) {
  /* Setup Vector Table */
  NVIC_SetupVectorTable();

  /* Init UART */
  mtk_os_hal_uart_ctlr_init(uart_port_num);
  printf("\nNeuroPilot-Micro Vision Demo\n");

  xTaskCreate(NN_Task, "NN Task", APP_STACK_SIZE_BYTES, NULL, 2, NULL);

  vTaskStartScheduler();
  for (;;)
    __asm__("wfi");
}
