/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "person_detect_model_data.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#ifdef BUILD_ARM_GCC
extern "C" void *__dso_handle __attribute__((weak));
#endif


const int tensor_arena_size = 73*1024;
uint8_t tensor_arena[tensor_arena_size] __attribute__((aligned(32)));

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

const int inputPersonTensorSize = 96 * 96;


extern "C" void person_detection_setup()
{
	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	model = ::tflite::GetModel(person_detect_model_data);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		error_reporter->Report(
			"Model provided is schema version %d not equal "
			"to supported version %d.\r\n",
			model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}
	error_reporter->Report("GetModel done, size %d bytes.\r\n", person_detect_model_data_len );

	static tflite::MicroMutableOpResolver<3> resolver;
	resolver.AddBuiltin(
		tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
		tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
	resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
		tflite::ops::micro::Register_CONV_2D());
	resolver.AddBuiltin(tflite::BuiltinOperator_AVERAGE_POOL_2D,
		tflite::ops::micro::Register_AVERAGE_POOL_2D());

	static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena,
		tensor_arena_size, error_reporter);
	interpreter = &static_interpreter;

	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		error_reporter->Report("AllocateTensors() failed.\r\n");
		return;	
	}

	model_input = interpreter->input(0);
	model_output = interpreter->output(0);
}

// 1: has person, 2: no person
extern "C" int person_detection_loop(uint8_t* input_buf)
{
	for (int i = 0; i < inputPersonTensorSize; ++i) {
		model_input->data.uint8[i] = input_buf[i];
	}

	if (kTfLiteOk != interpreter->Invoke()) {
		error_reporter->Report("Invoke failed.");
		return -1;
	}

	uint8_t person_score = model_output->data.uint8[1];  // kPersonIndex
	uint8_t no_person_score = model_output->data.uint8[2];  // kNotAPersonIndex
	error_reporter->Report("person score:%d no person score %d", person_score,
                         no_person_score);

	if(person_score > no_person_score) {
		return 1;
	}
	else {
		return 2;
	}
}






