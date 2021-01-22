#ifndef __DYNAMIC_SCRIPT_H__
#define __DYNAMIC_SCRIPT_H__

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/micro/micro_allocator.h"

#ifdef DYNAMIC_SCRIPT_SUPPORT

namespace tflite {

typedef struct DynamicScene {
  uintptr_t src;
  uintptr_t dest;
  size_t size;
  int layer_index; // The action time
  int tensor_index;
  DynamicLoadAction action;
  DynamicScene& operator=(DynamicScene& a) {
    this->src = a.src;
    this->dest = a.dest;
    this->size = a.size;
    this->layer_index = a.layer_index;
    this->tensor_index = a.tensor_index;
    this->action = a.action;
    return *this;
  }
} DynamicScene;


inline void swapScene(DynamicScene &a, DynamicScene &b) {
  DynamicScene temp;
  temp = a;
  a = b;
  b = temp;
}

typedef struct DynamicScript {
  size_t size;
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
    __GNUC_MINOR__ >= 1
  DynamicScene scene[0];
#else
  DynamicScene scene[];
#endif
} DynamicScript;

class DynamicScriptCreator {
public:
  DynamicScriptCreator(uint8_t *script_arena, size_t size);
  DynamicScriptCreator(void) {return; };

  TfLiteStatus Init(uint8_t *script_arena, size_t size);

  TfLiteStatus LoadScript(DynamicScript &script);

  TfLiteStatus AppendScene(int layer, DynamicLoadAction action,
                           int tensor_index, size_t offset, size_t size);

  void DumpScriptStructure(ErrorReporter *error_reporter);
  void DumpScript(ErrorReporter *error_reporter);
  void DumpScript(ErrorReporter *error_reporter, size_t operators_size,
                  size_t max_buffer_size,
                  NodeAndRegistration *node_and_registrations);

  DynamicScene *LookUpScene(int tensor_idx);

  size_t script_size(void) const {
    return (script_ != nullptr) ? script_->size : 0;
  };

  size_t script_memory_bytes(void) const {
    return (script_ != nullptr) ?
          (sizeof(DynamicScript) + sizeof(DynamicScene) * script_->size) : 0;
  };

  DynamicScript *GetScript(void) { return script_; };
  TfLiteStatus ClearScript(void);

  void sort(void);
private:
  DynamicScript *script_ = nullptr;
  size_t script_max_size_;
  uint8_t *script_arena_;
};

} // namespace tflite

#endif // DYNAMIC_SCRIPT_SUPPORT
#endif //__DYNAMIC_SCRIPT_H__