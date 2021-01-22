#ifndef __CIFAR10_MAIN_FUNCTIONS_H__
#define __CIFAR10_MAIN_FUNCTIONS_H__
#ifdef  __cplusplus
extern  "C" {
#endif

#include <stdint.h>

int cifar10_setup(void);

int cifar10_invoke(const uint8_t *input);

int cifar10_loop(void);

#ifdef  __cplusplus
}
#endif
#endif