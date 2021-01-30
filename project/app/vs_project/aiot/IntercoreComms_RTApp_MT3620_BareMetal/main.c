/* Copyright (c) Microsoft Corporation. All rights reserved.
   Licensed under the MIT License. */

// This sample C application for the real-time core demonstrates intercore communications by
// sending a message to a high-level application every second, and printing out any received
// messages.
//
// It demontrates the following hardware
// - UART (used to write a message via the built-in UART)
// - mailbox (used to report buffer sizes and send / receive events)
// - timer (used to send a message to the HLApp)

#include <ctype.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>

#include "FreeRTOS.h"
#include "logical-dpc.h"
#include "logical-intercore.h"

#include "mt3620-baremetal.h"
#include "mt3620-intercore.h"
#include "mt3620-timer.h"
#include <semphr.h>

#define DATALENGTH 22050
int16_t AdcData[DATALENGTH] = { 0 };
int rear = 0, full = 0;

void insert(int16_t data) {
    AdcData[rear] = data;
    rear++;
    rear %= DATALENGTH;
    if (!full && rear == 0) {
        full = 1;
    }
}

int16_t get() {
    rear++;
    rear %= DATALENGTH;
    return AdcData[rear];
}

static const char* label[] = { "NO BREATH", "BREATH", "CAUGH", "SPEAK" };
extern void emergency_detect_setup();
extern int emergency_detect_loop(int16_t* input_buf);

extern uint32_t StackTop; // &StackTop == end of TCM

static IntercoreComm icc;

static const uint32_t sendTimerIntervalMs = 1000;

static _Noreturn void DefaultExceptionHandler(void);
static void HandleSendTimerIrq(void);
static void HandleSendTimerDeferred(void);

_Noreturn void RTCoreMain(void);

/* Hook for "stack over flow". */
void vApplicationStackOverflowHook(TaskHandle_t xTask, char* pcTaskName) {
    printf("%s: %s\n", __func__, pcTaskName);
}

/* Hook for "memory allocation failed". */
void vApplicationMallocFailedHook(void) {
    printf("%s\n", __func__);
}

// ARM DDI0403E.d SB1.5.2-3
// From SB1.5.3, "The Vector table must be naturally aligned to a power of two whose alignment
// value is greater than or equal to (Number of Exceptions supported x 4), with a minimum alignment
// of 128 bytes.". The array is aligned in linker.ld, using the dedicated section ".vector_table".

// The exception vector table contains a stack pointer, 15 exception handlers, and an entry for
// each interrupt.
#define INTERRUPT_COUNT 100 // from datasheet
#define EXCEPTION_COUNT (16 + INTERRUPT_COUNT)
#define INT_TO_EXC(i_) (16 + (i_))
const uintptr_t ExceptionVectorTable[EXCEPTION_COUNT] __attribute__((section(".vector_table")))
__attribute__((used)) = {
    [0] = (uintptr_t)&StackTop,                // Main Stack Pointer (MSP)
    [1] = (uintptr_t)RTCoreMain,               // Reset
    [2] = (uintptr_t)DefaultExceptionHandler,  // NMI
    [3] = (uintptr_t)DefaultExceptionHandler,  // HardFault
    [4] = (uintptr_t)DefaultExceptionHandler,  // MPU Fault
    [5] = (uintptr_t)DefaultExceptionHandler,  // Bus Fault
    [6] = (uintptr_t)DefaultExceptionHandler,  // Usage Fault
    [11] = (uintptr_t)DefaultExceptionHandler, // SVCall
    [12] = (uintptr_t)DefaultExceptionHandler, // Debug monitor
    [14] = (uintptr_t)DefaultExceptionHandler, // PendSV
    [15] = (uintptr_t)DefaultExceptionHandler, // SysTick

    [INT_TO_EXC(0)] = (uintptr_t)DefaultExceptionHandler,
    [INT_TO_EXC(1)] = (uintptr_t)MT3620_Gpt_HandleIrq1,
    [INT_TO_EXC(2)... INT_TO_EXC(10)] = (uintptr_t)DefaultExceptionHandler,
    [INT_TO_EXC(11)] = (uintptr_t)MT3620_HandleMailboxIrq11,
    [INT_TO_EXC(12)... INT_TO_EXC(INTERRUPT_COUNT - 1)] = (uintptr_t)DefaultExceptionHandler};

// If the applications end up in this function then an unexpected exception has occurred.
static _Noreturn void DefaultExceptionHandler(void)
{
    for (;;) {
        // empty.
    }
}

// Runs in IRQ context and schedules HandleSendTimerDeferred to run later.
static void HandleSendTimerIrq(void)
{
    static CallbackNode cbn = {.enqueued = false, .cb = HandleSendTimerDeferred};
    EnqueueDeferredProc(&cbn);
}

// Queued by HandleSendTimerIrq. Sends a message to the HLApp.
static void HandleSendTimerDeferred(void)
{
    static int iter = 0;
    // The component ID for IntercoreComms_HighLevelApp.
    static const ComponentId hlAppId = {.data1 = 0x25025d2c,
                                        .data2 = 0x66da,
                                        .data3 = 0x4448,
                                        .data4 = {0xba, 0xe1, 0xac, 0x26, 0xfc, 0xdd, 0x36, 0x27}};

    // The number cycles from "00" to "99".
    static char txMsg[] = "rt-app-to-hl-app-00";
    const size_t txMsgLen = sizeof(txMsg);

    IntercoreResult icr = IntercoreSend(&icc, &hlAppId, txMsg, sizeof(txMsg) - 1);
    if (icr != Intercore_OK) {
    }

    txMsg[txMsgLen - 3] = '0' + (iter / 10);
    txMsg[txMsgLen - 2] = '0' + (iter % 10);
    iter = (iter + 1) % 100;

    MT3620_Gpt_LaunchTimerMs(TimerGpt0, sendTimerIntervalMs, HandleSendTimerIrq);
}

// Runs with interrupts enabled. Retrieves messages from the inbound buffer
// and prints their sender ID, length, and content (hex and text).
static void HandleReceivedMessageDeferred(void)
{
    emergency_detect_setup();
    for (;;) {
        ComponentId sender;
        uint8_t rxData[2];
        size_t rxDataSize = sizeof(rxData);

        IntercoreResult icr = IntercoreRecv(&icc, &sender, rxData, &rxDataSize);

        // Return if read all messages in buffer.
        if (icr == Intercore_Recv_NoBlockSize) {
            return;
        }

        // Return if an error occurred.
        if (icr != Intercore_OK) {
            return;
        }

        int16_t rowData;
        rowData = (rxData[0] << 8) | rxData[1];
        insert(rowData);
        if (full) {
            int16_t inputData[DATALENGTH];
            for (int16_t i = 0; i < DATALENGTH; i++) {
                inputData[i] = AdcData[(i + rear) % DATALENGTH];
            }
            uint8_t result = emergency_detect_loop(inputData);
        }
    }
}

_Noreturn void RTCoreMain(void)
{
    // The debugger will not connect until shortly after the application has started running.
    // To use the debugger with code which runs at application startup, change the initial value
    // of b from true to false, run the app, break into the app with a debugger, and set b to true.
    volatile bool b = true;
    while (!b) {
        // empty.
    }

    // SCB->VTOR = ExceptionVectorTable
    WriteReg32(SCB_BASE, 0x08, (uint32_t)ExceptionVectorTable);

    MT3620_Gpt_Init();

    IntercoreResult icr = SetupIntercoreComm(&icc, HandleReceivedMessageDeferred);
    if (icr != Intercore_OK) {
    } else {
        MT3620_Gpt_LaunchTimerMs(TimerGpt0, sendTimerIntervalMs, HandleSendTimerIrq);
    }

    for (;;) {
        InvokeDeferredProcs();
        __asm__("wfi");
    }
}
