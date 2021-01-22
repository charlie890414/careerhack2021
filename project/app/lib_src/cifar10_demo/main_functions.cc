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

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "cifar10_model_data.h"
#include "main_functions.h"

#ifdef BUILD_ARM_GCC
extern "C" void *__dso_handle __attribute__((weak));
#endif

const int tensor_arena_size = 45 * 1024;
uint8_t tensor_arena[tensor_arena_size] __attribute__((aligned(32)));

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

const int inputTensorSize = 32 * 32 * 3;

extern "C" int cifar10_setup()
{
	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	model = tflite::GetModel(cifar10_model_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		error_reporter->Report(
			"Model provided is schema version %d not equal "
			"to supported version %d.\r\n",
			model->version(), TFLITE_SCHEMA_VERSION);
		return kTfLiteError;
	}
	error_reporter->Report("GetModel %s done, size %d bytes.\r\n",
			get_model_file_name(), cifar10_model_tflite_len);

	static tflite::MicroMutableOpResolver<4> resolver;
	resolver.AddBuiltin(
		tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
		tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
	resolver.AddBuiltin(
		tflite::BuiltinOperator_CONV_2D,
		tflite::ops::micro::Register_CONV_2D());
	resolver.AddBuiltin(
		tflite::BuiltinOperator_FULLY_CONNECTED,
		tflite::ops::micro::Register_FULLY_CONNECTED());
	resolver.AddBuiltin(
		tflite::BuiltinOperator_MAX_POOL_2D,
		tflite::ops::micro::Register_MAX_POOL_2D());

	static tflite::MicroInterpreter static_interpreter(model, resolver,
		tensor_arena, tensor_arena_size, error_reporter);
	interpreter = &static_interpreter;

	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		error_reporter->Report("AllocateTensors() failed.\r\n");
		return kTfLiteError;
	}

	model_input = interpreter->input(0);
	model_output = interpreter->output(0);

	return kTfLiteOk;
}

extern "C" int cifar10_invoke(const uint8_t *input)
{
	uint8_t top[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	const char* cifar10_label[] = {"Plane", "Car", "Bird", "Cat",
				       "Deer", "Dog", "Frog", "Horse",
				       "Ship", "Truck" };

	for (int i = 0; i < inputTensorSize; i++) {
		model_input->data.uint8[i] = input[i];
	}

	TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
		error_reporter->Report("Invoke failed.\r\n");
		return -1;
	}

	for (int i = 0; i < 3; i++) {
		for (int j = i + 1; j < 10; j++){
			if (model_output->data.uint8[j] > model_output->data.uint8[i]) {
				std::swap(model_output->data.uint8[j], model_output->data.uint8[i]);
				std::swap(top[j], top[i]);
			}
		}
	}

	error_reporter->Report("top 1 result: %s, %d\r\n", cifar10_label[top[0]], model_output->data.uint8[0]);
	error_reporter->Report("top 2 result: %s, %d\r\n", cifar10_label[top[1]], model_output->data.uint8[1]);
	error_reporter->Report("top 3 result: %s, %d\r\n", cifar10_label[top[2]], model_output->data.uint8[2]);

	return top[0];
}

extern "C" int cifar10_loop(void)
{
	return kTfLiteOk;
}
