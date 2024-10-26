#include <cstddef>
#include <cstdint>

#include "../common/consts.hpp"
#include "../gpu-common/gpu-utils.cuh"
#include "fls-test-kernels-global.cuh"

namespace fls {
namespace gpu {
namespace test {

template <typename T>
void bitunpack(const T *__restrict in, T *__restrict out, const size_t count,
               const int32_t value_bit_width) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_blocks = n_vecs;
  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(count);

  kernels::fls::global::test::bitunpack<T, 1, 4>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(device_in.get(), device_out.get(),
                                              value_bit_width);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);
}
template <typename T>
void unffor(const T *__restrict in, T *__restrict out, const size_t count,
            const int32_t value_bit_width, const T *__restrict base_p) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_blocks = n_vecs;

  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(count);
  GPUArray<T> device_base_p(1, base_p);

  kernels::fls::global::test::unffor<T, 1, utils::get_values_per_lane<T>()>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(device_in.get(), device_out.get(),
                                              value_bit_width,
                                              device_base_p.get());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);
}

} // namespace test
} // namespace gpu
} // namespace fls

template void fls::gpu::test::bitunpack<uint8_t>(const uint8_t *__restrict in,
                                                 uint8_t *__restrict out,
                                                 const size_t count,
                                                 const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::unffor<uint8_t>(const uint8_t *__restrict in,
                                              uint8_t *__restrict out,
                                              const size_t count,
                                              const int32_t value_bit_width,
                                              const uint8_t *__restrict base_p);
template void fls::gpu::test::unffor<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint16_t *__restrict base_p);
template void fls::gpu::test::unffor<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint32_t *__restrict base_p);
template void fls::gpu::test::unffor<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint64_t *__restrict base_p);
