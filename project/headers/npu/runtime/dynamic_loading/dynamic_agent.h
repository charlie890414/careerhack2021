#ifndef __dynamic_agent_H__
#define __dynamic_agent_H__

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "dynamic_script.h"

// For MT3620
#define DEFAULT_DYNAMIC_ARENA_SIZE (5 * 1024)
#define DEFAULT_FINE_GRAINED_SIZE (4 * 1024)
// For other
//#define DEFAULT_DYNAMIC_ARENA_SIZE (10 * 1024)
//#define DEFAULT_FINE_GRAINED_SIZE (2 * 1024)

#define DEFAULT_FINE_GRAINED_OPCODE_LIST { \
    BuiltinOperator_CONV_2D, \
  }

namespace tflite {

class DynamicAgent {
public:
  DynamicAgent(TfLiteContext &context, ErrorReporter *error_reporter,
          uint8_t* tensor_arena, size_t tensor_arena_size):
    context_(&context), error_reporter_(error_reporter),
#ifdef DYNAMIC_SCRIPT_SUPPORT
    script_creator_(dynamic_arena_, DEFAULT_DYNAMIC_ARENA_SIZE),
#endif
    tensor_arena_(tensor_arena),
    tensor_arena_size_(tensor_arena_size) { return ; };

  DynamicAgent(void) { return ; };

  TfLiteStatus Init(TfLiteContext &context, ErrorReporter *error_reporter,
          uint8_t* tensor_arena, size_t tensor_arena_size);

  TfLiteStatus Finalize(NodeAndRegistration *node_and_registrations,
                                     size_t operators_size);
  TfLiteStatus MicroRuntimePreprocess(int layer_index);
  TfLiteStatus MicroRuntimePostprocess(int layer_index);

  TfLiteStatus RegisterCache(uint8_t *cache, size_t size);

  TfLiteStatus EnableDynamicLoad(void);
  TfLiteStatus DisableDynamicLoad(void);

#ifdef DYNAMIC_SCRIPT_SUPPORT
  TfLiteStatus SetScriptSupport(bool script_support);
  bool IsScriptSupport(void) const { return script_support_; }
  TfLiteStatus SetScriptEnable(bool script_support);
#endif // DYNAMIC_SCRIPT_SUPPORT


  TfLiteStatus SetCacheEnable(bool flag);
  bool IsCacheEnable(void) const { return cache_enable_; }
  bool IsCacheAvailable(void) const { return cache_available_; }

  TfLiteStatus SetFineGrainedEnable(bool flag);
  TfLiteStatus SetSizeOriented(bool flag);

  bool IsFineGrainedOpcode(const int32_t builtin_code);

  size_t GetDynamicArenaSize(void) { return dynamic_arena_size_; }
  TfLiteStatus SetDynamicArenaSize(size_t size);

  size_t GetFineGrainedSize(void) { return fine_grained_size_; }
  TfLiteStatus SetFineGrainedSize(size_t size);

  void PrintConfig(void);

protected:

#ifdef INTERNAL_MEMORY_ARRAY
  TfLiteStatus MutateRuntimeArray(void);
#endif // INTERNAL_MEMORY_ARRAY

#ifdef DYNAMIC_SCRIPT_SUPPORT
  TfLiteStatus CreateSimpleScript(void);
  TfLiteStatus FinalizeDynamicScript(void);
  TfLiteStatus UnsetDynamicScript(void);

  TfLiteStatus ScriptPreprocess(int layer_index);
  TfLiteStatus ScriptPostprocess(int layer_index);
#endif

  DynamicLoadAction DynamicLoadPolicy(NodeAndRegistration &node_registration,
                                      DataIndex index, size_t &data_size);

  TfLiteStatus DefaultPreprocess(int layer_index);
  TfLiteStatus DefaultPostprocess(int layer_index);

private:
  ErrorReporter *error_reporter_ = nullptr;
  uint8_t *tensor_arena_ = nullptr;
  size_t tensor_arena_size_;
  TfLiteContext *context_;
  NodeAndRegistration *node_and_registrations_ = nullptr;
  size_t operators_size_;

  bool dynamic_enable_ = true;
  bool fine_grained_enable_ = true;
  bool size_oriented_enable_ = true;

  size_t fine_grained_size_ = DEFAULT_FINE_GRAINED_SIZE;
  size_t dynamic_buffer_size_ = 0;
  uint8_t *dynamic_buffer_ = nullptr;

  size_t dynamic_cache_size_ = 0;
  uint8_t *dynamic_cache_ = nullptr;
  bool cache_enable_ = false;
  bool cache_available_ = false;

#ifdef DYNAMIC_SCRIPT_SUPPORT
  bool script_support_ = true;
  bool script_enable_ = true;
  class DynamicScriptCreator script_creator_;
#endif // DYNAMIC_SCRIPT_SUPPORT

#ifdef INTERNAL_MEMORY_ARRAY
  bool runtime_array_support_ = true;
  size_t array_arena_bytes_ = 0;
  size_t array_arena_size_;
  uint8_t *array_arena_;
#endif // INTERNAL_MEMORY_ARRAY

#ifndef MICRO_VECTOR
  int32_t fine_grained_opcode_list_[10] = DEFAULT_FINE_GRAINED_OPCODE_LIST;
#else
  std::vector<int> fine_grained_opcode_list_ = DEFAULT_FINE_GRAINED_OPCODE_LIST;
#endif //MICRO_VECTOR

  size_t dynamic_arena_bytes_ = 0;
  size_t dynamic_arena_size_ = DEFAULT_DYNAMIC_ARENA_SIZE;
  uint8_t dynamic_arena_[DEFAULT_DYNAMIC_ARENA_SIZE]
                                    __attribute__((aligned(DATA_ALIGN))) = {0};
};

} // namespce tflite

#endif //__DYNAMIC_LOAD_H__