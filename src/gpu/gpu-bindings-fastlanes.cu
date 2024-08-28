#include <cstddef>
#include <cstdint>

#include "../consts.h"
#include "fastlanes-global.cuh"
#include "gpu-bindings-fastlanes.h"
#include "gpu-utils.cuh"

template <typename T>
void bitunpack(const T *__restrict in, T *__restrict out, const size_t count,
               const int32_t value_bit_width, const bool use_reader) {
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

  if (!use_reader) {
    bitunpack_with_function_global<T, 1, utils::get_values_per_lane<T>()>
        <<<n_blocks, utils::get_n_lanes<T>()>>>(device_in, device_out,
                                                value_bit_width);
  } else {
    bitunpack_with_reader_global<T, 1, utils::get_values_per_lane<T>()>
        <<<n_blocks, utils::get_n_lanes<T>()>>>(device_in, device_out,
                                                value_bit_width);
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  CUDA_SAFE_CALL(
      cudaMemcpy(out, device_out, decoded_size, cudaMemcpyDeviceToHost));
}

template <typename T>
void gpu::bitunpack_with_function(const T *__restrict in, T *__restrict out,
                                  const size_t count,
                                  const int32_t value_bit_width) {
  bitunpack(in, out, count, value_bit_width, false);
}

template <typename T>
void gpu::bitunpack_with_reader(const T *__restrict in, T *__restrict out,
                                const size_t count,
                                const int32_t value_bit_width) {
  bitunpack(in, out, count, value_bit_width, true);
}

template void gpu::bitunpack_with_function<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void
gpu::bitunpack_with_reader<uint8_t>(const uint8_t *__restrict in,
                                    uint8_t *__restrict out, const size_t count,
                                    const int32_t value_bit_width);
template void gpu::bitunpack_with_function<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void gpu::bitunpack_with_reader<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void gpu::bitunpack_with_function<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void gpu::bitunpack_with_reader<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void gpu::bitunpack_with_function<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void gpu::bitunpack_with_reader<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
