#include <cstddef>
#include <cstdint>

#include "../consts.h"
#include "fastlanes-global.cuh"
#include "gpu-bindings-fastlanes.h"
#include "gpu-utils.cuh"

namespace gpu {
template <typename T>
void bitunpack(const T *__restrict in, T *__restrict out, const size_t count,
               const int32_t value_bit_width) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_blocks = n_vecs;
  const auto encoded_size = (count * static_cast<size_t>(value_bit_width)) / 8;
  const auto decoded_size = count * sizeof(T);

  T *device_in = nullptr;
  CUDA_SAFE_CALL(
      cudaMalloc(reinterpret_cast<void **>(&device_in), encoded_size));
  CUDA_SAFE_CALL(
      cudaMemcpy(device_in, in, encoded_size, cudaMemcpyHostToDevice));

  T *device_out = nullptr;
  CUDA_SAFE_CALL(
      cudaMalloc(reinterpret_cast<void **>(&device_out), decoded_size));

  bitunpack_global<T, 1, utils::get_values_per_lane<T>()>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(device_in, device_out,
                                              value_bit_width);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  CUDA_SAFE_CALL(
      cudaMemcpy(out, device_out, decoded_size, cudaMemcpyDeviceToHost));
}
} // namespace gpu

template void gpu::bitunpack<uint8_t>(const uint8_t *__restrict in,
                                      uint8_t *__restrict out,
                                      const size_t count,
                                      const int32_t value_bit_width);
template void gpu::bitunpack<uint16_t>(const uint16_t *__restrict in,
                                       uint16_t *__restrict out,
                                       const size_t count,
                                       const int32_t value_bit_width);
template void gpu::bitunpack<uint32_t>(const uint32_t *__restrict in,
                                       uint32_t *__restrict out,
                                       const size_t count,
                                       const int32_t value_bit_width);
template void gpu::bitunpack<uint64_t>(const uint64_t *__restrict in,
                                       uint64_t *__restrict out,
                                       const size_t count,
                                       const int32_t value_bit_width);
