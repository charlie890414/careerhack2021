#include "powerdown_lib.h"

// This constant defines the maximum time (in seconds) the device can be in powerdown mode. A value
// of less than 2 seconds will cause the device to resume from powerdown immediately, behaving like
// a reboot.

void TriggerPowerdown(const unsigned int powerdownResidencyTime)
{
    // Put the device in the powerdown mode
    int result = PowerManagement_ForceSystemPowerDown(powerdownResidencyTime);
    if (result != 0) {
        Log_Debug("Error PowerManagement_ForceSystemPowerDown: %s (%d).\n", strerror(errno), errno);
        return;
    }
}