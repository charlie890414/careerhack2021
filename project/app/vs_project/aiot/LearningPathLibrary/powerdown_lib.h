
#ifndef POWERDOWN_LIB_H
#define POWERDOWN_LIB_H

#include "applibs_versions.h"
#include <applibs/log.h>
#include <applibs/powermanagement.h>

#include <errno.h>
#include <string.h>

void TriggerPowerdown(const unsigned int powerdownResidencyTime);

#endif