#pragma once

#include "hw/azure_sphere_learning_path.h"
#include "imu_temp_pressure.h"
#include "light_sensor.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

typedef struct
{
	float temperature;
	float humidity;
	float pressure;
	int light;
	int CO2;
	int dB;
} LP_ENVIRONMENT;

bool lp_readTelemetry(LP_ENVIRONMENT* environment);
bool lp_initializeDevKit(void);
bool lp_closeDevKit(void);
