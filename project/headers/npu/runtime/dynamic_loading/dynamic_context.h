#ifndef __DYNAMIC_CONTEXT_H__
#define __DYNAMIC_CONTEXT_H__

#include <stddef.h>
#ifdef __cplusplus
    extern "C" {
#endif

// dynamic loading version 0.3
#define MICRO_RUNTIME

// dynamic loading version 0.3 - Normal DMA

// dynamic loading version 0.4 - INTERNAL_MEMORY_ARRAY + Normal DMA
//#define INTERNAL_MEMORY_ARRAY

// dynamic loading version 0.5 - DYNAMIC_SCRIPT_SUPPORT + DVFS + DMA
//#define DYNAMIC_SCRIPT_SUPPORT

#define DYNAMIC_CACHE_SUPPORT

bool IsFineGrainedOPSupport(void);

typedef enum DataIndex {
  InputIndex = 0,
  WeightsIndex = 1,
  BiasIndex = 2,
  DataIndexNum,
} DataIndex;

typedef enum DynamicLoadAction {
  DL_NotLoad,
  DL_NotSupport,
  DL_Error,
  DL_Load,
  DL_FineGrained,
  DL_DVFS_Enable,
  DL_DVFS_Disable,
  DL_Script_Disable,
} DynamicLoadAction;

inline bool IsLoadAction(DynamicLoadAction action) {
  return (action == DL_Load) || (action == DL_FineGrained);
}

inline const char* EnumNameDynamicLoadAction(DynamicLoadAction action) {
  static const char * const names[8] = {
    "DL_NotLoad",
    "DL_NotSupport",
    "DL_Error",
    "DL_Load",
    "DL_FineGrained",
    "DL_DVFS_Enable",
    "DL_DVFS_Disable",
    "DL_Script_Disable",
  };
  return names[(int)action];
}

typedef struct DynamicLoadInfo {
  void *current_extaddr;
  DynamicLoadAction current_action;
  void *dynamic_buffer;
  size_t max_size;
  bool fine_grained_flag;
} DynamicLoadInfo;

typedef struct DynamicContext {
  DynamicLoadInfo dl_info[DataIndexNum];
} DynamicContext;

#ifdef __cplusplus
    }
#endif
#endif // __DYNAMIC_CONTEXT_H__
