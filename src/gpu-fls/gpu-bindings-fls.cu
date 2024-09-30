#include <cstddef>
#include <cstdint>

#include "../common/consts.hpp"
#include "fls-global.cuh"
#include "gpu-bindings-fls.hpp"
#include "../gpu-common/gpu-utils.cuh"

namespace gpu {
template <typename T>
void bitunpack(const T *__restrict in, T *__restrict out, const size_t count,
               const int32_t value_bit_width) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_blocks = n_vecs;
  const auto encoded_count = value_bit_width == 0 ? 1 : (count * static_cast<size_t>(value_bit_width)) / (8 *sizeof(T));

  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(count);

  bitunpack_global<T, 1, utils::get_values_per_lane<T>()>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(device_in.get(),
                                              device_out.get(),
                                              value_bit_width);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);
}
template <typename T>
void unffor(const T *__restrict in, T *__restrict out, const size_t count,
            const int32_t value_bit_width, const T *__restrict base_p) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_blocks = n_vecs;

  const auto encoded_count = value_bit_width == 0 ? 1 : (count * static_cast<size_t>(value_bit_width)) / (8 *sizeof(T));


  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(count);
  GPUArray<T> device_base_p(1, base_p);

  unffor_global<T, 1, utils::get_values_per_lane<T>()>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(
          device_in.get(), device_out.get(),
          value_bit_width, device_base_p.get());
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
