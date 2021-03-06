#include "emergency-detect.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#ifdef BUILD_ARM_GCC
extern "C" void *__dso_handle __attribute__((weak));
#endif

const int tensor_arena_size = 73 * 1024;
uint8_t tensor_arena[tensor_arena_size];

tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;

TfLiteTensor *model_input = nullptr;
TfLiteTensor *model_output = nullptr;

const int emergency_detect_input_size = 22050;
const char *emergency_detect_classes[] = { "NO BREATH", "BREATH", "CAUGH", "SPEAK" };

extern "C" void emergency_detect_setup() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = ::tflite::GetModel(output_emergency_detect_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\r\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  error_reporter->Report("GetModel done, size %d bytes.\r\n", output_emergency_detect_tflite_len);

  static tflite::AllOpsResolver resolver;
                      
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

extern "C" int emergency_detect_loop(int16_t *input_buf) {
  for (int i = 0; i < emergency_detect_input_size; ++i) {
    model_input->data.i16[i] = input_buf[i];
  }

  if (kTfLiteOk != interpreter->Invoke()) {
    error_reporter->Report("Invoke failed.");
    return -1;
  }

  uint8_t max_score = 0;
  int max_score_index = 0;
  for (int i = 0; i < 4; ++i) {
    uint8_t score = model_output->data.uint8[i];
    error_reporter->Report("%s score: %d", emergency_detect_classes[i], score);

    if (score > max_score) max_score_index = i;
  }

  return max_score_index;
}
