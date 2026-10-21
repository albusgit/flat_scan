/* LiteRT GPU delegate check of a .tflite on an Android device or emulator (tools/litert/gpu_check.sh runs it).

     gpu_check LIB_DIR MODEL [IN.f32 OUT.f32] [--cpu | --gl]
       LIB_DIR: folder of libtensorflowlite_jni.so + libtensorflowlite_gpu_jni.so (the LiteRT 1.4.0 AARs' jni/<abi>)
       IN.f32: the input tensor's bytes (float32 NHWC); OUT.f32: written with the output tensor's bytes
       --cpu: no delegate (the LiteRT CPU kernels), for comparison
       --gl: the delegate's OpenGL ES backend only (TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY). Needed where there is
             no OpenCL (the emulator): LiteRT 1.4.0's GPU delegate then crashes in TfLiteInterpreterCreate (SIGSEGV,
             strlen of a null dlerror() while it formats "Failed to load OpenCL library with dlopen: ")

   Loads both libraries with dlopen and calls the TFLite C API through dlsym, as core/src/infer.cpp does (contract
   C3). The GPU delegate gets full precision: is_precision_loss_allowed 0, inference_priority1 MAX_PRECISION (fp32
   compute). The interpreter applies the delegate when it is created and LiteRT logs (stderr and logcat, tag
   tflite) "Replacing N out of M node(s) with delegate (TfLiteGpuDelegateV2) node, yielding K partitions": all
   nodes in 1 partition means no op falls back to the CPU. Then, with IN.f32, one Invoke and the output written.
   Exit status: 0 ok, 1 usage / load error, 2 interpreter or delegate creation failed, 3 invoke failed. */
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteTensor TfLiteTensor;
typedef struct TfLiteDelegate TfLiteDelegate;

/* tflite/delegates/gpu/delegate_options.h (LiteRT 1.4.0); pad: room for fields a build may add at the end */
typedef struct {
  int32_t is_precision_loss_allowed;
  int32_t inference_preference;
  int32_t inference_priority1;
  int32_t inference_priority2;
  int32_t inference_priority3;
  int64_t experimental_flags;
  int32_t max_delegated_partitions;
  const char* serialization_dir;
  const char* model_token;
  char pad[256];
} GpuOptions;

#define FN(ret, name, args) static ret(*name) args
FN(TfLiteModel*, TfLiteModelCreateFromFile, (const char*));
FN(TfLiteInterpreterOptions*, TfLiteInterpreterOptionsCreate, (void));
FN(void, TfLiteInterpreterOptionsAddDelegate, (TfLiteInterpreterOptions*, TfLiteDelegate*));
FN(void, TfLiteInterpreterOptionsSetErrorReporter, (TfLiteInterpreterOptions*, void (*)(void*, const char*, va_list), void*));
FN(void, TfLiteInterpreterOptionsSetNumThreads, (TfLiteInterpreterOptions*, int32_t));
FN(TfLiteInterpreter*, TfLiteInterpreterCreate, (const TfLiteModel*, const TfLiteInterpreterOptions*));
FN(int, TfLiteInterpreterAllocateTensors, (TfLiteInterpreter*));
FN(int, TfLiteInterpreterInvoke, (TfLiteInterpreter*));
FN(TfLiteTensor*, TfLiteInterpreterGetInputTensor, (const TfLiteInterpreter*, int32_t));
FN(const TfLiteTensor*, TfLiteInterpreterGetOutputTensor, (const TfLiteInterpreter*, int32_t));
FN(size_t, TfLiteTensorByteSize, (const TfLiteTensor*));
FN(int32_t, TfLiteTensorNumDims, (const TfLiteTensor*));
FN(int32_t, TfLiteTensorDim, (const TfLiteTensor*, int32_t));
FN(int, TfLiteTensorCopyFromBuffer, (TfLiteTensor*, const void*, size_t));
FN(int, TfLiteTensorCopyToBuffer, (const TfLiteTensor*, void*, size_t));
FN(GpuOptions, TfLiteGpuDelegateOptionsV2Default, (void));
FN(TfLiteDelegate*, TfLiteGpuDelegateV2Create, (const GpuOptions*));

static void reporter(void* user, const char* fmt, va_list args) {
  (void)user;
  printf("tflite error: ");
  vprintf(fmt, args);
  printf("\n");
}

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void shape(const char* what, const TfLiteTensor* t) {
  printf("%s [", what);
  for (int i = 0; i < TfLiteTensorNumDims(t); ++i) printf(i ? ",%d" : "%d", TfLiteTensorDim(t, i));
  printf("] %zu bytes\n", TfLiteTensorByteSize(t));
}

#define SYM(lib, name)                                            \
  if (!(*(void**)&name = dlsym(lib, #name))) {                    \
    printf("missing symbol %s\n", #name);                         \
    return 1;                                                     \
  }

int main(int argc, char** argv) {
  int cpu = argc > 1 && strcmp(argv[argc - 1], "--cpu") == 0;
  int gl = argc > 1 && strcmp(argv[argc - 1], "--gl") == 0;
  int n = argc - cpu - gl;
  if (n != 3 && n != 5) {
    printf("usage: gpu_check LIB_DIR MODEL [IN.f32 OUT.f32] [--cpu | --gl]\n");
    return 1;
  }
  char path[1024];
  snprintf(path, sizeof path, "%s/libtensorflowlite_jni.so", argv[1]);
  void* lt = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
  snprintf(path, sizeof path, "%s/libtensorflowlite_gpu_jni.so", argv[1]);
  void* gpu = lt ? dlopen(path, RTLD_NOW | RTLD_GLOBAL) : NULL;
  if (!lt || !gpu) {
    printf("dlopen failed: %s\n", dlerror());
    return 1;
  }
  SYM(lt, TfLiteModelCreateFromFile) SYM(lt, TfLiteInterpreterOptionsCreate) SYM(lt, TfLiteInterpreterOptionsAddDelegate)
  SYM(lt, TfLiteInterpreterOptionsSetErrorReporter) SYM(lt, TfLiteInterpreterOptionsSetNumThreads)
  SYM(lt, TfLiteInterpreterCreate) SYM(lt, TfLiteInterpreterAllocateTensors) SYM(lt, TfLiteInterpreterInvoke)
  SYM(lt, TfLiteInterpreterGetInputTensor) SYM(lt, TfLiteInterpreterGetOutputTensor) SYM(lt, TfLiteTensorByteSize)
  SYM(lt, TfLiteTensorNumDims) SYM(lt, TfLiteTensorDim) SYM(lt, TfLiteTensorCopyFromBuffer) SYM(lt, TfLiteTensorCopyToBuffer)
  SYM(gpu, TfLiteGpuDelegateOptionsV2Default) SYM(gpu, TfLiteGpuDelegateV2Create)

  TfLiteModel* model = TfLiteModelCreateFromFile(argv[2]);
  if (!model) {
    printf("cannot load model %s\n", argv[2]);
    return 1;
  }
  TfLiteInterpreterOptions* opt = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetErrorReporter(opt, reporter, NULL);
  TfLiteInterpreterOptionsSetNumThreads(opt, 4);
  if (!cpu) {
    GpuOptions g = TfLiteGpuDelegateOptionsV2Default();
    g.is_precision_loss_allowed = 0;
    g.inference_preference = 1;   /* SUSTAINED_SPEED */
    g.inference_priority1 = 1;    /* MAX_PRECISION: fp32 compute */
    g.inference_priority2 = 0;    /* AUTO */
    g.inference_priority3 = 0;
    if (gl) g.experimental_flags |= 1 << 2;   /* TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY */
    printf("gpu options: precision_loss %d priority %d/%d/%d flags %lld max_partitions %d\n", g.is_precision_loss_allowed,
           g.inference_priority1, g.inference_priority2, g.inference_priority3, (long long)g.experimental_flags,
           g.max_delegated_partitions);
    TfLiteDelegate* d = TfLiteGpuDelegateV2Create(&g);
    if (!d) {
      printf("TfLiteGpuDelegateV2Create failed\n");
      return 2;
    }
    TfLiteInterpreterOptionsAddDelegate(opt, d);
  }
  fflush(stdout);
  double t0 = now_ms();
  TfLiteInterpreter* it = TfLiteInterpreterCreate(model, opt);
  if (!it || TfLiteInterpreterAllocateTensors(it) != 0) {
    printf("interpreter creation failed (%s)\n", cpu ? "cpu" : "gpu delegate");
    return 2;
  }
  printf("interpreter created (%s) in %.0f ms\n", cpu ? "cpu" : "gpu delegate", now_ms() - t0);
  TfLiteTensor* in = TfLiteInterpreterGetInputTensor(it, 0);
  const TfLiteTensor* out = TfLiteInterpreterGetOutputTensor(it, 0);
  shape("input", in);
  shape("output", out);
  if (n == 5) {
    size_t ni = TfLiteTensorByteSize(in), no = TfLiteTensorByteSize(out);
    void* bi = malloc(ni);
    void* bo = malloc(no);
    FILE* f = fopen(argv[3], "rb");
    if (!f || fread(bi, 1, ni, f) != ni) {
      printf("cannot read %zu bytes from %s\n", ni, argv[3]);
      return 1;
    }
    fclose(f);
    TfLiteTensorCopyFromBuffer(in, bi, ni);
    for (int k = 0; k < 2; ++k) {   /* the first run includes the GPU's warm-up */
      t0 = now_ms();
      if (TfLiteInterpreterInvoke(it) != 0) {
        printf("invoke failed\n");
        return 3;
      }
      printf("invoke %d: %.0f ms\n", k, now_ms() - t0);
    }
    TfLiteTensorCopyToBuffer(out, bo, no);
    f = fopen(argv[4], "wb");
    if (!f || fwrite(bo, 1, no, f) != no) {
      printf("cannot write %s\n", argv[4]);
      return 1;
    }
    fclose(f);
    printf("output written to %s\n", argv[4]);
  }
  return 0;
}
