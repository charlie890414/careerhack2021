#ifndef __CIFAR10_MODEL_DATA_H__
#define __CIFAR10_MODEL_DATA_H__
// Example MNIST classification model,for use with TFlite Micro

// Model tflite flatbuffer.
extern const unsigned char cifar10_model_tflite[];

// Length of model tflite flatbuffer.
extern const unsigned int cifar10_model_tflite_len;

char* get_model_file_name(void);

#endif  // __CIFAR10_MODEL_DATA_H__
