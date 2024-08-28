#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#define CUDA_SAFE_CALL(call)                                                   \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__,    \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <typename T> class GPUArray {
private:
  size_t size;
  T *device_p = nullptr;

  void allocate() {
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&device_p), size));
  }

public:
  GPUArray(const size_t a_size) {
    size = a_size;
    allocate();
  }

  GPUArray(const size_t a_size, const T *host_p) {
    size = a_size;
    allocate();
    CUDA_SAFE_CALL(cudaMemcpy(device_p, host_p, size, cudaMemcpyHostToDevice));
  }

  void copy_to_host(T *host_p) {
    CUDA_SAFE_CALL(cudaMemcpy(host_p, device_p, size, cudaMemcpyDeviceToHost));
  }

  T *get_pointer() { return device_p; }

  ~GPUArray() {
    if (device_p != nullptr) {
      CUDA_SAFE_CALL(cudaFree(device_p));
    }
  }
};

#endif // GPU_UTILS_H
