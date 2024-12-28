#include <cstddef>
#include <cstdint>

#include "../common/consts.hpp"
#include "../gpu-common/gpu-utils.cuh"
#include "fls-test-kernels-global.cuh"

namespace fls {
namespace gpu {
namespace test {

template <typename T, unsigned UNPACK_N_VECTORS = 1>
void bitunpack(const T *__restrict in, T *__restrict out, const size_t count,
               const int32_t value_bit_width) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_threads = utils::get_n_lanes<T>();
  const auto n_blocks = n_vecs / UNPACK_N_VECTORS;
  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(count);

  kernels::fls::global::test::bitunpack<T, UNPACK_N_VECTORS, 1>
      <<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                                value_bit_width);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);
}

template <typename T, unsigned UNPACK_N_VECTORS = 1>
void bitunpack_with_state(const T *__restrict in, T *__restrict out,
                          const size_t count, const int32_t value_bit_width) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_threads = utils::get_n_lanes<T>();
  const auto n_blocks = n_vecs / UNPACK_N_VECTORS;
  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(count);

  kernels::fls::global::test::bitunpack_with_state<T, UNPACK_N_VECTORS, 1>
      <<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
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

template void fls::gpu::test::bitunpack<uint8_t, 1>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint16_t, 1>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint32_t, 1>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint64_t, 1>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

template void fls::gpu::test::bitunpack<uint8_t, 2>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint16_t, 2>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint32_t, 2>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint64_t, 2>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

template void fls::gpu::test::bitunpack<uint8_t, 4>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint16_t, 4>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint32_t, 4>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack<uint64_t, 4>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

template void fls::gpu::test::bitunpack_with_state<uint8_t, 1>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack_with_state<uint16_t, 1>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack_with_state<uint32_t, 1>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void fls::gpu::test::bitunpack_with_state<uint64_t, 1>(
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
