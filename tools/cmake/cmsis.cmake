cmake_minimum_required(VERSION 3.8)

message("### setting library cmsis ###")

set(PREBUILTS_DIR ${CMAKE_SOURCE_DIR}/prebuilts)

include_directories(
	headers
	headers/cmsis/CMSIS/Core/Include
	headers/cmsis/CMSIS/NN/Include
	headers/cmsis/CMSIS/DSP/Include 
	)


file(GLOB_RECURSE EXT_LIB_FILES
	source/cmsis/*.c
	source/cmsis/*.h
)
add_library(cmsis STATIC ${EXT_LIB_FILES})

list(APPEND ALL_EXT_LIBS cmsis)