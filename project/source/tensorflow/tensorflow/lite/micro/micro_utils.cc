/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_utils.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/op_macros.h"

#if ENABLE_TIME_MEASUREMENT
#include "tensorflow/lite/micro/micro_time.h"
#include <string.h>

#define TIME_SLOT (200) // space usage (4 + 1 + 4 + 1) * TIME_SLOT bytes
int start_index = 0;
uint32_t  start_time[TIME_SLOT] = {0};
const char* start_labels[TIME_SLOT] = {0};

int end_index = 0;
uint32_t  end_time[TIME_SLOT] = {0};
const char* end_labels[TIME_SLOT] = {0};

uint32_t get_time(void) {
  return tflite::GetCurrentTimeTicks();
}

/* keep start time stamp and labels */
void start_record_time(const char* str)
{
  if (start_index >= TIME_SLOT)
    return;

  start_labels[start_index] = str;
  start_time[start_index] = get_time();
  start_index++;
}

/* keep end time stamp and labels */
void end_record_time(const char* str)
{
  if (end_index >= TIME_SLOT)
    return;

  end_time[end_index] = get_time();
  end_labels[end_index] = str;
  end_index++;
}

/* Report the time measurement */
void time_reporter(tflite::ErrorReporter* reporter)
{
  if (start_index >= TIME_SLOT || end_index >= TIME_SLOT)
    reporter->Report("Index meets maximum TIME_SLOT %d. may lose some recodes.",
                     TIME_SLOT);

  /* sort the end_time to meet the order in start_label
   * The situation may be like:
   *   start_time  [ 3, 5, 7, 9, 12, 15]
   *   start_label [ a, b, b, c, d,  c] (allow to use the same label)
   *
   *   end_time  [6, 8, 11, 17, 19, 20]
   *   end_label [b, b, c,  c,  d,  a]
   *
   * sort end_time by start_label order
   *   sorted_end_time [ 20, 6, 8, 11, 19, 17]
   *   start_time      [ 3,  5, 7, 9,  12, 15]
   *   start_label     [ a,  b, b, c,  d,  c]
   * We can easily calculate the execution time and find overlapping time
   * intervals.
   * */
  uint32_t  sorted_end_time[TIME_SLOT] = {0};
  for (int i = 0; i < end_index; i++) {
    int first_match = 0;
    for (int j = 0; j < start_index && end_time[i] >= start_time[j]; j++) {
      if (strcmp(end_labels[i], start_labels[j]) == 0 &&
          sorted_end_time[j] == 0) {
        first_match = j;
        break;
      }
    }
    sorted_end_time[first_match] = end_time[i];
  }

  for (int i = 0; i < start_index; i++) {
    int depth = 0;

    for (int j = i - 1; j >= 0; j--) {
      if (start_time[i] >= start_time[j] &&
          sorted_end_time[i] <= sorted_end_time[j]) {
        /* time interval i overlaps on j time interval */
        if (!(start_time[i] == sorted_end_time[i] &&
              start_time[i] == sorted_end_time[j])) {
            /* TODO: A best way to determine overlapping
             * This is a workaround:
             * Not increase the depth for the situation that a time interval
             * (start_time[i], sorted_end_time[i]) is 0 time and the
             * start_time[i] equals to sorted_end_time[j],
             * because we can not determine if this to time intervals are
             * overlapping or not.
             * for example:
             *   start_labels    [ a,   b,  c]
             *   start_time      [ 17, 20, 20]
             *   sorted_end_time [ 20, 20, 20]
             * There will be serval situations, such as:
             *
             *   start_record_time("a");
             *   start_record_time("b");
             *   start_record_time("c");
             *   end_record_time("c");
             *   end_record_time("b");
             *   end_record_time("a");
             * or
             *   start_record_time("a");
             *   end_record_time("a");
             *   start_record_time("b");
             *   end_record_time("b");
             *   start_record_time("c");
             *   end_record_time("c");
             * and more...
             * */
          depth++;
        }
      }
    }

    char depth_prefix[32] = { 0 };
    int p = 0;
    for (;depth > 0 && p < 32; depth--) {
      depth_prefix[p++] = '|';
      depth_prefix[p++] = ' ';
    }
    depth_prefix[p++] = '/';
    depth_prefix[p] = '\0';

    reporter->Report("%s %s - time :%d", depth_prefix,
              start_labels[i], sorted_end_time[i] - start_time[i]);
  }

  /* reset */
  start_index = 0;
  end_index = 0;
}
#else // ENABLE_TIME_MEASUREMENT
void start_record_time(const char* str)
{
  return;
}

void end_record_time(const char* str)
{
  return;
}

void time_reporter(tflite::ErrorReporter* reporter)
{
  return;
}
#endif // ENABLE_TIME_MEASUREMENT

namespace tflite {

namespace {

static const uint8_t kAsymmetricUInt8Min = 0;
static const uint8_t kAsymmetricUInt8Max = UINT8_MAX;
static const uint8_t kSymmetricUInt8Min = 1;
static const uint8_t kSymmetricUInt8Max = UINT8_MAX;
static const int8_t kAsymmetricInt8Min = INT8_MIN;
static const int8_t kAsymmetricInt8Max = INT8_MAX;
static const int kSymmetricInt8Scale = kAsymmetricInt8Max;

static const int16_t kAsymmetricInt16Min = INT16_MIN;
static const int16_t kAsymmetricInt16Max = INT16_MAX;
static const int kSymmetricInt16Scale = kAsymmetricInt16Max;

static const int32_t kAsymmetricInt32Max = INT32_MAX;
static const int kSymmetricInt32Scale = kAsymmetricInt32Max;

}  // namespace

int ElementCount(const TfLiteIntArray& dims) {
  int result = 1;
  for (int i = 0; i < dims.size; ++i) {
    result *= dims.data[i];
  }
  return result;
}

// Converts a float value into an unsigned eight-bit quantized value.
uint8_t FloatToAsymmetricQuantizedUInt8(const float value, const float scale,
                                        const int zero_point) {
  int32_t result = round(value / scale) + zero_point;
  if (result < kAsymmetricUInt8Min) {
    result = kAsymmetricUInt8Min;
  }
  if (result > kAsymmetricUInt8Max) {
    result = kAsymmetricUInt8Max;
  }
  return result;
}

uint8_t FloatToSymmetricQuantizedUInt8(const float value, const float scale) {
  int32_t result = round(value / scale);
  if (result < kSymmetricUInt8Min) {
    result = kSymmetricUInt8Min;
  }
  if (result > kSymmetricUInt8Max) {
    result = kSymmetricUInt8Max;
  }
  return result;
}

int8_t FloatToAsymmetricQuantizedInt8(const float value, const float scale,
                                      const int zero_point) {
  int32_t result = round(value / scale) + zero_point;
  if (result < kAsymmetricInt8Min) {
    result = kAsymmetricInt8Min;
  }
  if (result > kAsymmetricInt8Max) {
    result = kAsymmetricInt8Max;
  }
  return result;
}

int16_t FloatToAsymmetricQuantizedInt16(const float value, const float scale,
                                        const int zero_point) {
  int32_t result = round(value / scale) + zero_point;
  if (result < kAsymmetricInt16Min) {
    result = kAsymmetricInt16Min;
  }
  if (result > kAsymmetricInt16Max) {
    result = kAsymmetricInt16Max;
  }
  return result;
}

int8_t FloatToSymmetricQuantizedInt8(const float value, const float scale) {
  return FloatToAsymmetricQuantizedInt8(value, scale, 0.0f);
}

int32_t FloatToSymmetricQuantizedInt32(const float value, const float scale) {
  float quantized = round(value / scale);
  if (static_cast<int>(quantized) > INT_MAX) {
    quantized = static_cast<float>(INT_MAX);
  } else if (quantized < INT_MIN) {
    quantized = static_cast<float> INT_MIN;
  }

  return static_cast<int>(quantized);
}

void AsymmetricQuantize(const float* input, int8_t* output, int num_elements,
                        float scale, int zero_point) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = FloatToAsymmetricQuantizedInt8(input[i], scale, zero_point);
  }
}

void AsymmetricQuantize(const float* input, uint8_t* output, int num_elements,
                        float scale, int zero_point) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = FloatToAsymmetricQuantizedUInt8(input[i], scale, zero_point);
  }
}

void AsymmetricQuantize(const float* input, int16_t* output, int num_elements,
                        float scale, int zero_point) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = FloatToAsymmetricQuantizedInt16(input[i], scale, zero_point);
  }
}

void SymmetricQuantize(const float* input, int32_t* output, int num_elements,
                       float scale) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = FloatToSymmetricQuantizedInt32(input[i], scale);
  }
}

void SymmetricPerChannelQuantize(const float* input, int32_t* output,
                                 int num_elements, int num_channels,
                                 float* scales) {
  int elements_per_channel = num_elements / num_channels;
  for (int i = 0; i < num_channels; i++) {
    for (int j = 0; j < elements_per_channel; j++) {
      output[i * elements_per_channel + j] = FloatToSymmetricQuantizedInt32(
          input[i * elements_per_channel + j], scales[i]);
    }
  }
}

void SignedSymmetricPerChannelQuantize(const float* values,
                                       TfLiteIntArray* dims,
                                       int quantized_dimension,
                                       int8_t* quantized_values,
                                       float* scaling_factors) {
  int input_size = ElementCount(*dims);
  int channel_count = dims->data[quantized_dimension];
  int per_channel_size = input_size / channel_count;

  int stride;
  int channel_stride;
  if (quantized_dimension == 0) {
    stride = 1;
    channel_stride = per_channel_size;
  } else if (quantized_dimension == 3) {
    stride = channel_count;
    channel_stride = 1;
  } else {
    TF_LITE_FATAL("quantized dimension must be 0 or 3");
  }

  // Calculate scales for each channel.
  for (int channel = 0; channel < channel_count; channel++) {
    float min = 0;
    float max = 0;

    for (int i = 0; i < per_channel_size; i++) {
      int idx = channel * channel_stride + i * stride;
      min = fminf(min, values[idx]);
      max = fmaxf(max, values[idx]);
    }
    scaling_factors[channel] =
        fmaxf(fabs(min), fabs(max)) / kSymmetricInt8Scale;
    for (int i = 0; i < per_channel_size; i++) {
      int idx = channel * channel_stride + i * stride;
      const int32_t quantized_value =
          static_cast<int32_t>(roundf(values[idx] / scaling_factors[channel]));
      // Clamp: just in case some odd numeric offset.
      quantized_values[idx] = fminf(
          kSymmetricInt8Scale, fmaxf(-kSymmetricInt8Scale, quantized_value));
    }
  }
}

void SignedSymmetricQuantize(const float* values, TfLiteIntArray* dims,
                             int8_t* quantized_values, float* scaling_factor) {
  int input_size = ElementCount(*dims);

  float min = 0;
  float max = 0;
  for (int i = 0; i < input_size; i++) {
    min = fminf(min, values[i]);
    max = fmaxf(max, values[i]);
  }
  *scaling_factor = fmaxf(fabs(min), fabs(max)) / kSymmetricInt8Scale;
  for (int i = 0; i < input_size; i++) {
    const int32_t quantized_value =
        static_cast<int32_t>(roundf(values[i] / *scaling_factor));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = fminf(kSymmetricInt8Scale,
                                fmaxf(-kSymmetricInt8Scale, quantized_value));
  }
}

void SignedSymmetricQuantize(const float* values, TfLiteIntArray* dims,
                             int16_t* quantized_values, float* scaling_factor) {
  int input_size = ElementCount(*dims);

  float min = 0;
  float max = 0;
  for (int i = 0; i < input_size; i++) {
    min = fminf(min, values[i]);
    max = fmaxf(max, values[i]);
  }
  *scaling_factor = fmaxf(fabs(min), fabs(max)) / kSymmetricInt16Scale;
  for (int i = 0; i < input_size; i++) {
    const int32_t quantized_value =
        static_cast<int32_t>(roundf(values[i] / *scaling_factor));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = fminf(kSymmetricInt16Scale,
                                fmaxf(-kSymmetricInt16Scale, quantized_value));
  }
}

void SignedSymmetricQuantize(const float* values, TfLiteIntArray* dims,
                             int32_t* quantized_values, float* scaling_factor) {
  int input_size = ElementCount(*dims);

  float min = 0;
  float max = 0;
  for (int i = 0; i < input_size; i++) {
    min = fminf(min, values[i]);
    max = fmaxf(max, values[i]);
  }

  *scaling_factor =
      fmaxf(fabs(min), fabs(max)) / static_cast<float>(kSymmetricInt32Scale);
  for (int i = 0; i < input_size; i++) {
    const int32_t quantized_value =
        static_cast<int32_t>(roundf(values[i] / *scaling_factor));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = fminf(
        static_cast<float>(kSymmetricInt32Scale),
        fmaxf(static_cast<float>(-kSymmetricInt32Scale), quantized_value));
  }
}

void SymmetricQuantize(const float* values, TfLiteIntArray* dims,
                       uint8_t* quantized_values, float* scaling_factor) {
  SignedSymmetricQuantize(values, dims,
                          reinterpret_cast<int8_t*>(quantized_values),
                          scaling_factor);
}

void SymmetricDequantize(const int8_t* values, const int size,
                         const float dequantization_scale,
                         float* dequantized_values) {
  for (int i = 0; i < size; ++i) {
    dequantized_values[i] = values[i] * dequantization_scale;
  }
}

}  // namespace tflite
