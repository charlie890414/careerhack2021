/* Copyright (c) Microsoft Corporation. All rights reserved.
   Licensed under the MIT License. */

// This sample C application for Azure Sphere sends messages to, and receives
// responses from, a real-time capable application. It sends a message every
// second and prints the message which was sent, and the response which was received.
//
// It uses the following Azure Sphere libraries
// - log (displays messages in the Device Output window during debugging)
// - application (establish a connection with a real-time capable application).
// - eventloop (system invokes handlers for timer events)

// Hardware definition
#include "hw/azure_sphere_learning_path.h"

#include <signal.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdbool.h>
#include <errno.h>
#include <unistd.h>

#include <sys/time.h>
#include <sys/socket.h>

#include <applibs/adc.h>
#include <applibs/log.h>
#include <applibs/application.h>

#include "eventloop_timer_utilities.h"

/// <summary>
/// Exit codes for this application. These are used for the
/// application exit code. They must all be between zero and 255,
/// where zero is reserved for successful termination.
/// </summary>
typedef enum {
    ExitCode_Success = 0,
    ExitCode_TermHandler_SigTerm = 1,
    ExitCode_TimerHandler_Consume = 2,
    ExitCode_SendMsg_Send = 3,
    ExitCode_SocketHandler_Recv = 4,
    ExitCode_Init_EventLoop = 5,
    ExitCode_Init_SendTimer = 6,
    ExitCode_Init_Connection = 7,
    ExitCode_Init_SetSockOpt = 8,
    ExitCode_Init_RegisterIo = 9,
    ExitCode_Main_EventLoopFail = 10,
    ExitCode_AdcTimerHandler_Consume = 11,
    ExitCode_AdcTimerHandler_Poll = 12,
    ExitCode_Init_AdcOpen = 13,
    ExitCode_Init_AdcPollTimer = 14,
    ExitCode_Init_GetBitCount = 15,
    ExitCode_Init_UnexpectedBitCount = 16,
} ExitCode;

static int sockFd = -1;
static int adcControllerFd = -1;

static const char* label[] = { "NO BREATH", "BREATH", "CAUGH", "SPEAK" };

static int sampleBitCount = -1;

static EventLoop *eventLoop = NULL;
static EventLoopTimer *sendTimer = NULL;
static EventLoopTimer* adcPollTimer = NULL;
static EventRegistration *socketEventReg = NULL;
static volatile sig_atomic_t exitCode = ExitCode_Success;

static const char rtAppComponentId[] = "005180bc-402f-4cb3-a662-72937dbcde47";

static void TerminationHandler(int signalNumber);
static void SendTimerEventHandler(EventLoopTimer *timer);
static void SendMessageToRTApp(uint16_t value);
static void SocketEventHandler(EventLoop *el, int fd, EventLoop_IoEvents events, void *context);
static ExitCode InitHandlers(void);
static void CloseHandlers(void);

/// <summary>
///     Signal handler for termination requests. This handler must be async-signal-safe.
/// </summary>
static void TerminationHandler(int signalNumber)
{
    // Don't use Log_Debug here, as it is not guaranteed to be async-signal-safe.
    exitCode = ExitCode_TermHandler_SigTerm;
}

/// <summary>
///     Handle send timer event by writing data to the real-time capable application.
/// </summary>
static void SendTimerEventHandler(EventLoopTimer *timer)
{
    if (ConsumeEventLoopTimerEvent(timer) != 0) {
        exitCode = ExitCode_TimerHandler_Consume;
        return;
    }

    //SendMessageToRTApp();
}

/// <summary>
///     Helper function for TimerEventHandler sends message to real-time capable application.
/// </summary>
static void SendMessageToRTApp(uint16_t value)
{
    // Send "hl-app-to-rt-app-%02d" message to RTApp, where the number cycles from 00 to 99.
    static int iter = 0;

    char msg[2];
    msg[0] = value >> 8;
    msg[1] = value & 0xFFFFFFFF;
    iter = (iter + 1) % 100;
    //Log_Debug("Sending: %s\n", msg);

    int bytesSent = send(sockFd, msg, strlen(msg), 0);
    if (bytesSent == -1) {
        Log_Debug("ERROR: Unable to send message: %d (%s)\n", errno, strerror(errno));
        exitCode = ExitCode_SendMsg_Send;
        return;
    }
}

/// <summary>
///     Handle socket event by reading incoming data from real-time capable application.
/// </summary>
static void SocketEventHandler(EventLoop *el, int fd, EventLoop_IoEvents events, void *context)
{
    // Read response from real-time capable application.
    // If the RTApp has sent more than 22050 bytes, then truncate.
    char rxBuf[32];
    int bytesReceived = recv(fd, rxBuf, sizeof(rxBuf), 0);

    if (bytesReceived == -1) {
        Log_Debug("ERROR: Unable to receive message: %d (%s)\n", errno, strerror(errno));
        exitCode = ExitCode_SocketHandler_Recv;
        return;
    }

    Log_Debug("Received %d bytes: ", bytesReceived);
    for (int i = 0; i < bytesReceived; ++i) {
        Log_Debug("%c", isprint(rxBuf[i]) ? rxBuf[i] : '.');
    }
    Log_Debug("\n");
}

static void AdcPollingEventHandler(EventLoopTimer* timer) {
    if (ConsumeEventLoopTimerEvent(timer) != 0) {
        exitCode = ExitCode_AdcTimerHandler_Consume;
        return;
    }

    uint32_t value;
    int result = ADC_Poll(adcControllerFd, ADC_CHANNEL1, &value);
    if (result == -1) {
        Log_Debug("ADC_Poll failed with error: %s (%d)\n", strerror(errno), errno);
        exitCode = ExitCode_AdcTimerHandler_Poll;
        return;
    }
    Log_Debug("%d\n", value);
    // SendMessageToRTApp(value);
}

/// <summary>
///     Set up SIGTERM termination handler and event handlers for send timer
///     and to receive data from real-time capable application.
/// </summary>
/// <returns>
///     ExitCode_Success if all resources were allocated successfully; otherwise another
///     ExitCode value which indicates the specific failure.
/// </returns>
static ExitCode InitHandlers(void)
{
    struct sigaction action;
    memset(&action, 0, sizeof(struct sigaction));
    action.sa_handler = TerminationHandler;
    sigaction(SIGTERM, &action, NULL);

    eventLoop = EventLoop_Create();
    if (eventLoop == NULL) {
        Log_Debug("Could not create event loop.\n");
        return ExitCode_Init_EventLoop;
    }

    adcControllerFd = ADC_Open(ADC_CONTROLLER0);
    if (adcControllerFd == -1) {
        Log_Debug("ADC_Open failed with error: %s (%d)\n", strerror(errno), errno);
        return ExitCode_Init_AdcOpen;
    }

    sampleBitCount = ADC_GetSampleBitCount(adcControllerFd, ADC_CONTROLLER0);
    if (sampleBitCount == -1) {
        Log_Debug("ADC_GetSampleBitCount failed with error : %s (%d)\n", strerror(errno), errno);
        return ExitCode_Init_GetBitCount;
    }

    if (sampleBitCount == 0) {
        Log_Debug("ADC_GetSampleBitCount returned sample size of 0 bits.\n");
        return ExitCode_Init_UnexpectedBitCount;
    }

    struct timespec adcCheckPeriod = { .tv_sec = 0, .tv_nsec = 1000000000/220500 };
    adcPollTimer = CreateEventLoopPeriodicTimer(eventLoop, &AdcPollingEventHandler, &adcCheckPeriod);
    if (adcPollTimer == NULL) {
        return ExitCode_Init_AdcPollTimer;
    }

    /*
    // Register a one second timer to send a message to the RTApp.
    static const struct timespec sendPeriod = {.tv_sec = 1, .tv_nsec = 0};
    sendTimer = CreateEventLoopPeriodicTimer(eventLoop, &SendTimerEventHandler, &sendPeriod);
    if (sendTimer == NULL) {
        return ExitCode_Init_SendTimer;
    }

    // Open a connection to the RTApp.
    sockFd = Application_Connect(rtAppComponentId);
    if (sockFd == -1) {
        Log_Debug("ERROR: Unable to create socket: %d (%s)\n", errno, strerror(errno));
        return ExitCode_Init_Connection;
    }

    // Set timeout, to handle case where real-time capable application does not respond.
    static const struct timeval recvTimeout = {.tv_sec = 5, .tv_usec = 0};
    int result = setsockopt(sockFd, SOL_SOCKET, SO_RCVTIMEO, &recvTimeout, sizeof(recvTimeout));
    if (result == -1) {
        Log_Debug("ERROR: Unable to set socket timeout: %d (%s)\n", errno, strerror(errno));
        return ExitCode_Init_SetSockOpt;
    }

    // Register handler for incoming messages from real-time capable application.
    socketEventReg = EventLoop_RegisterIo(eventLoop, sockFd, EventLoop_Input, SocketEventHandler, NULL);

    if (socketEventReg == NULL) {
        Log_Debug("ERROR: Unable to register socket event: %d (%s)\n", errno, strerror(errno));
        return ExitCode_Init_RegisterIo;
    }
    */

    return ExitCode_Success;
}

/// <summary>
///     Closes a file descriptor and prints an error on failure.
/// </summary>
/// <param name="fd">File descriptor to close</param>
/// <param name="fdName">File descriptor name to use in error message</param>
static void CloseFdAndPrintError(int fd, const char *fdName)
{
    if (fd >= 0) {
        int result = close(fd);
        if (result != 0) {
            Log_Debug("ERROR: Could not close fd %s: %s (%d).\n", fdName, strerror(errno), errno);
        }
    }
}

/// <summary>
///     Clean up the resources previously allocated.
/// </summary>
static void CloseHandlers(void)
{
    DisposeEventLoopTimer(sendTimer);
    EventLoop_UnregisterIo(eventLoop, socketEventReg);
    EventLoop_Close(eventLoop);

    Log_Debug("Closing file descriptors.\n");
    CloseFdAndPrintError(sockFd, "Socket");
}

int main(void)
{
    // Log_Debug("High-level intercore comms application\n");
    // Log_Debug("Sends data to, and receives data from a real-time capable application.\n");

    exitCode = InitHandlers();

    while (exitCode == ExitCode_Success) {
        EventLoop_Run_Result result = EventLoop_Run(eventLoop, -1, true);
        // Continue if interrupted by signal, e.g. due to breakpoint being set.
        if (result == EventLoop_Run_Failed && errno != EINTR) {
            exitCode = ExitCode_Main_EventLoopFail;
        }
    }

    CloseHandlers();
    Log_Debug("Application exiting.\n");
    return exitCode;
}
