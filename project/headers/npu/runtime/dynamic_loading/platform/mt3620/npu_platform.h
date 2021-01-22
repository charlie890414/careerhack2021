#ifndef __NPU_PLATFORM_H__
#define __NPU_PLATFORM_H__

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#define DYNAMIC_AGENT_VERSION (3)

#ifndef ROUND_DOWN
#define ROUND_DOWN(x, s) ((x) & ~((s)-1))
#endif
#ifndef ROUND_UP
#define ROUND_UP(x, s) (((x) + (s) - 1) & ~((s)-1))
#endif

#define DATA_ALIGN (32)

/* The memory copy function is dependent on platform */
void* PlatMemoryCopy(void *dest, const void *src, size_t size);

/* LoadFromExternal method is dependent on platform */
void* LoadFromExternal(void *dest, const void *src, size_t size);

/* IsExternalRegion method is dependent on platform */
bool IsExternalRegion(const uintptr_t addr);

#endif //__NPU_PLATFORM_H__
