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

#include "mnist_demo_model.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#ifdef BUILD_ARM_GCC
extern "C" void *__dso_handle __attribute__((weak));
#endif


const int tensor_arena_size = 22 * 1024;
uint8_t tensor_arena[tensor_arena_size] __attribute__((aligned(32)));


tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

const int inputTensorSize = 28 * 28;

extern "C" void mnist_setup()
{
	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	model = ::tflite::GetModel(mnist_dense_model_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		error_reporter->Report(
			"Model provided is schema version %d not equal "
			"to supported version %d.\r\n",
			model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}
	error_reporter->Report("GetModel done, size %d bytes.\r\n",
		mnist_dense_model_tflite_len );

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

	static tflite::MicroInterpreter static_interpreter(model, resolver,
		tensor_arena, tensor_arena_size, error_reporter);
	interpreter = &static_interpreter;

	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		error_reporter->Report("AllocateTensors() failed.\r\n");
		return;
	}

	model_input = interpreter->input(0);
	model_output = interpreter->output(0);

}

extern "C" int mnist_invoke(uint8_t* input_buf)
{
	int res_idx = 0;
	float res_val = 0.0f;

	for (int d = 0; d < inputTensorSize; ++d) {
		model_input->data.f[d] = (float)(input_buf[d])/255;
	}

	TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
		error_reporter->Report("Invoke failed.\r\n");
		return -1;
	}

	for(int i=0; i<10; i++) {
		if(model_output->data.f[i] > res_val) {
			res_val = model_output->data.f[i];
			res_idx = i;
		}
	}

	return res_idx;
}


