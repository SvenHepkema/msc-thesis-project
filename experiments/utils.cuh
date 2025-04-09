#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#define CUDA_SAFE_CALL(call)                                                   \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__,    \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <typename T> T *allocate_array(const size_t size) {
  T *device_ptr;
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&device_ptr), size));
  return device_ptr;
}

template <typename T> T *allocate_array(const size_t size, const T *host_ptr) {
  T *device_ptr;

  size_t allocation_size = size * sizeof(T);
  CUDA_SAFE_CALL(
      cudaMalloc(reinterpret_cast<void **>(&device_ptr), allocation_size));

  CUDA_SAFE_CALL(cudaMemcpy(device_ptr, host_ptr, allocation_size,
                            cudaMemcpyHostToDevice));

  return device_ptr;
}

template <typename T> void free_device_pointer(T *device_ptr) {
  CUDA_SAFE_CALL(cudaFree(device_ptr));
}
