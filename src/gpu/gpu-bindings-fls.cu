#include <cstddef>
#include <cstdint>

#include "../consts.h"
#include "fls-global.cuh"
#include "gpu-bindings-fls.hpp"
#include "gpu-utils.cuh"

namespace gpu {
template <typename T>
void bitunpack(const T *__restrict in, T *__restrict out, const size_t count,
               const int32_t value_bit_width) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_blocks = n_vecs;
  const auto encoded_size = (count * static_cast<size_t>(value_bit_width)) / 8;
  const auto decoded_size = count * sizeof(T);

  GPUArray<T> device_in(encoded_size, in);
  GPUArray<T> device_out(decoded_size);

  bitunpack_global<T, 1, utils::get_values_per_lane<T>()>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(device_in.get_pointer(),
                                              device_out.get_pointer(),
                                              value_bit_width);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);
}
template <typename T>
void unffor(const T *__restrict in, T *__restrict out, const size_t count,
            const int32_t value_bit_width, const T *__restrict base_p) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_blocks = n_vecs;
  const auto encoded_size = (count * static_cast<size_t>(value_bit_width)) / 8;
  const auto decoded_size = count * sizeof(T);

  GPUArray<T> device_in(encoded_size, in);
  GPUArray<T> device_out(decoded_size);
  GPUArray<T> device_base_p(sizeof(T), base_p);

  unffor_global<T, 1, utils::get_values_per_lane<T>()>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(
          device_in.get_pointer(), device_out.get_pointer(),
          value_bit_width, device_base_p.get_pointer());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);
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
template void gpu::unffor<uint8_t>(const uint8_t *__restrict in,
                                   uint8_t *__restrict out, const size_t count,
                                   const int32_t value_bit_width,
                                   const uint8_t *__restrict base_p);
template void gpu::unffor<uint16_t>(const uint16_t *__restrict in,
                                    uint16_t *__restrict out,
                                    const size_t count,
                                    const int32_t value_bit_width,
                                    const uint16_t *__restrict base_p);
template void gpu::unffor<uint32_t>(const uint32_t *__restrict in,
                                    uint32_t *__restrict out,
                                    const size_t count,
                                    const int32_t value_bit_width,
                                    const uint32_t *__restrict base_p);
template void gpu::unffor<uint64_t>(const uint64_t *__restrict in,
                                    uint64_t *__restrict out,
                                    const size_t count,
                                    const int32_t value_bit_width,
                                    const uint64_t *__restrict base_p);
