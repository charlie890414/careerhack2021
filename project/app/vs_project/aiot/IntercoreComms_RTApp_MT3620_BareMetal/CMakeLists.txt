# This code is based on a sample from Microsoft (see license below),
# with modifications made by MediaTek.
# Modified version of CMakeLists.txt from Microsoft Azure Sphere sample code:
# https://github.com/Azure/azure-sphere-samples/blob/master/Samples/HelloWorld/HelloWorld_RTApp_MT3620_BareMetal/CMakeLists.txt

#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License.

cmake_minimum_required(VERSION 3.10)
project(IntercoreComms_RTApp_MT3620_BareMetal C CXX)
azsphere_configure_tools(TOOLS_REVISION "20.07")

add_compile_definitions(OSAI_FREERTOS)
set(CMAKE_CXX_FLAGS "-Wno-reorder -Os -g -std=c++11 -fno-rtti -fpermissive -mlittle-endian -mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -fno-exceptions -ffunction-sections -fdata-sections -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS -DTF_LITE_USE_GLOBAL_MAX -DTF_LITE_USE_GLOBAL_MIN -DNDEBUG -DTF_LITE_STATIC_MEMORY -DBUILD_ARM_GCC  -DNEUROPILOT_MICRO")

add_link_options(-specs=nano.specs -specs=nosys.specs)
set(infer_model "EMERGENCY_DETECT")  # PERSON_DETECTION_DEMO, CIFAR10_DEMO, EMERGENCY_DETECT


if(${infer_model} STREQUAL "PERSON_DETECTION_DEMO")
    add_compile_definitions(PERSON_DETECTION_DEMO)

    # Executable
    add_executable(${PROJECT_NAME}
                   ./main.c
                   logical-intercore.c logical-dpc.c mt3620-intercore.c mt3620-uart-poll.c mt3620-timer.c 
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_gpio.c
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_uart.c
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_mbox.c
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_mbox_shared_mem.c
 	               ../../../lib_src/person_detection_demo/person_detect_model_data.cc
	               ../../../lib_src/person_detection_demo/main.cc)

elseif(${infer_model} STREQUAL "CIFAR10_DEMO")
    add_compile_definitions(CIFAR10_DEMO)

    # Executable
    add_executable(${PROJECT_NAME}
                   ./main.c
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_gpio.c
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_uart.c
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_mbox.c
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_mbox_shared_mem.c
                   ../../../lib_src/cifar10_demo/main.cc
                   ../../../lib_src/cifar10_demo/main_functions.cc
                   ../../../lib_src/cifar10_demo/cifar10_model_data.cc)

elseif(${infer_model} STREQUAL "EMERGENCY_DETECT")
    add_compile_definitions(EMERGENCY_DETECT)

    # Executable
    add_executable(${PROJECT_NAME}
                   ./main.c
                   logical-intercore.c logical-dpc.c mt3620-intercore.c mt3620-uart-poll.c mt3620-timer.c 
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_gpio.c
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_uart.c
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_mbox.c
                   ../../../../source/RTCORE_OS_HAL/src/os_hal_mbox_shared_mem.c
 	               ../../../lib_src/emergency_detect/emergency-detect.cc
	               ../../../lib_src/emergency_detect/main.cc)

endif()

# Include Folders
include_directories(${PROJECT_NAME} PUBLIC
                        ../../../../source/RTCORE_OS_HAL/inc
                        ./)
target_include_directories(${PROJECT_NAME} PUBLIC
                           ../../../../source/RTCORE_OS_HAL/inc
                           ./)
# include NPu headers
target_include_directories(${PROJECT_NAME} PUBLIC
    ../../../../headers/
    ../../../../headers/npu/kernels
	../../../../headers/npu/runtime
	../../../../headers/npu/runtime/dynamic_loading
	../../../../headers/npu/runtime/dynamic_loading/platform/mt3620
    ../../../../headers/cmsis/CMSIS/Core/Include
	../../../../headers/cmsis/CMSIS/NN/Include
	../../../../headers/cmsis/CMSIS/DSP/Include 
	../../../../third_party/flatbuffers/include
	../../../../third_party/gemmlowp
	../../../../third_party/kissfft
	../../../../third_party/ruy
    ../../../../source/tensorflow
 )

# Libraries
add_library( libtensorflow-microlite STATIC IMPORTED)
SET_TARGET_PROPERTIES( libtensorflow-microlite PROPERTIES IMPORTED_LOCATION ../../../../../../prebuilts/lib/libtensorflow-microlite.a)

add_library( libcmsis STATIC IMPORTED)
SET_TARGET_PROPERTIES( libcmsis PROPERTIES IMPORTED_LOCATION ../../../../../../prebuilts/lib/libcmsis.a)

set(OSAI_FREERTOS 1)
add_subdirectory(../../../../../MT3620_M4_Driver ./lib/MT3620_M4_Driver)
target_link_libraries(${PROJECT_NAME}
                      MT3620_M4_Driver
                      libtensorflow-microlite
                      libcmsis
                      stdc++ supc++ m c gcc nosys)

# Linker, Image
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_DEPENDS ${CMAKE_SOURCE_DIR}/linker.ld)
azsphere_target_add_image_package(${PROJECT_NAME})
