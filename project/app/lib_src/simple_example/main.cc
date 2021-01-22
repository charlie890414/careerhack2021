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

#include "mnist_model.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#ifdef BUILD_ARM_GCC
extern "C" void *__dso_handle __attribute__((weak));
#endif

namespace simple_mnist {
// Prepare tensor arena
const int tensor_arena_size = 22 * 1024;
uint8_t tensor_arena[tensor_arena_size] __attribute__((aligned(32)));

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Input 28 x 28 image
const int inputTensorSize = 28 * 28;
// Output 10 scores for number 0-9
const int outputTensorSize = 10;
}

using namespace simple_mnist;

// Export functions
extern "C" void model_setup(void);
extern "C" void model_loop(void);
extern "C" int model_invoke(uint8_t* input_buf);

void model_setup(void)
{
	// Create an MicroErrorReporter for logging
	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	// Load the model
	model = ::tflite::GetModel(mnist_model_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		TF_LITE_REPORT_ERROR(error_reporter,
			"Model provided is schema version %d not equal "
			"to supported version %d.\r\n",
			model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}
	TF_LITE_REPORT_ERROR(error_reporter,
		"GetModel done, size %d bytes.\r\n", mnist_model_tflite_len );

#ifdef LINUX_DEBUG
	// Only use AllOpsResolver when testing on Linux x86
	static tflite::AllOpsResolver resolver;
#else
	static tflite::MicroMutableOpResolver<7> resolver;
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
		tflite::BuiltinOperator_AVERAGE_POOL_2D,
		tflite::ops::micro::Register_AVERAGE_POOL_2D());
	resolver.AddBuiltin(
		tflite::BuiltinOperator_SOFTMAX,
		tflite::ops::micro::Register_SOFTMAX());
	resolver.AddBuiltin(
		tflite::BuiltinOperator_QUANTIZE,
		tflite::ops::micro::Register_QUANTIZE());
	resolver.AddBuiltin(
		tflite::BuiltinOperator_DEQUANTIZE,
		tflite::ops::micro::Register_DEQUANTIZE());
#endif

	// Create an interpreter
	static tflite::MicroInterpreter static_interpreter(model, resolver,
		tensor_arena, tensor_arena_size, error_reporter);
	interpreter = &static_interpreter;

	// Allocate resources
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter,
				     "AllocateTensors() failed.\r\n");
		return;
	}

	model_input = interpreter->input(0);
	model_output = interpreter->output(0);

}

int model_invoke(uint8_t* input_buf)
{
	for (int i = 0; i < inputTensorSize; ++i) {
		model_input->data.f[i] = (float)(input_buf[i])/255;
	}

	// Run the model
	TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.\r\n");
		return -1;
	}

	uint8_t max_val = std::numeric_limits<float>::min();
	int max_idx = 0;

	for(int i = 0; i < outputTensorSize; i++) {
		if (model_output->data.f[i] > max_val) {
			max_val = model_output->data.f[i];
			max_idx = i;
		}
	}

	return max_idx;
}

void model_loop(void) {
	uint8_t tmp[inputTensorSize] = {0};
	int res;

	res = model_invoke(tmp);
	TF_LITE_REPORT_ERROR(error_reporter, "Get prediction %d.\r\n", res);
	return ;
}

#ifdef LINUX_DEBUG
int main(int argc, char* argv[]) {
	int i = 3;

	model_setup();
	while (i-- > 0) {
		model_loop();
	}

	return 0;
}
#endif