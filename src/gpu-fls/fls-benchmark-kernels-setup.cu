#include <cstddef>
#include <cstdint>

#include "../common/consts.hpp"
#include "../gpu-common/gpu-utils.cuh"
#include "fls-benchmark-kernels-global.cuh"

namespace fls {
namespace gpu {
namespace bench {

template <typename T>
void query_baseline_contains_zero(const T *__restrict in, T *__restrict out,
                                  const size_t count) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_vectors_per_block = 2;
  const auto n_blocks = n_vecs / n_vectors_per_block;
  const auto n_threads = utils::get_n_lanes<T>() * 2;

  GPUArray<T> device_in(count, in);
  GPUArray<T> device_out(1);

  kernels::fls::global::bench::query_baseline_contains_zero<T, 1>
      <<<n_blocks, n_threads>>>(device_in.get(), device_out.get());

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);

  if (*out != 1) {
    *out = 0;
  }
}

template <typename T>
void query_old_fls_contains_zero(const T *__restrict in, T *__restrict out,
                            const size_t count, const int32_t value_bit_width
                            ) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  constexpr auto UNPACK_N_VECTORS = 1;
	const auto n_warps_per_block = 1;
  const auto n_vectors_per_block = n_warps_per_block * UNPACK_N_VECTORS;
  const auto n_blocks = n_vecs / n_vectors_per_block;
  const auto n_threads = utils::get_n_lanes<T>() * n_warps_per_block;

  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(1);

	kernels::fls::global::bench::query_old_fls_contains_zero<T, UNPACK_N_VECTORS, 32>
			<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
																value_bit_width);

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);

  if (*out != 1) {
    *out = 0;
  }
}

template <typename T>
void query_bp_contains_zero(const T *__restrict in, T *__restrict out,
                            const size_t count, const int32_t value_bit_width,
                            const int32_t unpack_n_values) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  constexpr auto UNPACK_N_VECTORS = 1;
  const auto n_vectors_per_block = 2 * UNPACK_N_VECTORS;
  const auto n_blocks = n_vecs / n_vectors_per_block;
  const auto n_threads = utils::get_n_lanes<T>() * 2;

  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(1);

  switch (unpack_n_values) {
  default:
  case 1:
    kernels::fls::global::bench::query_bp_contains_zero<T, UNPACK_N_VECTORS, 1>
        <<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                                  value_bit_width);
    break;
    /*
case 2:
kernels::fls::global::bench::query_bp_contains_zero<
T, UNPACK_N_VECTORS, 2>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width);
break;
case 4:
kernels::fls::global::bench::query_bp_contains_zero<
T, UNPACK_N_VECTORS, 4>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width);
break;
case 8:
kernels::fls::global::bench::query_bp_contains_zero<
T, UNPACK_N_VECTORS, 8>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width);
break;
case 16:
kernels::fls::global::bench::query_bp_contains_zero<
T, UNPACK_N_VECTORS, 16>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width);
break;
case 32:
kernels::fls::global::bench::query_bp_contains_zero<
T, UNPACK_N_VECTORS, 32>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width);
break;
case 64:
kernels::fls::global::bench::query_bp_contains_zero<
T, UNPACK_N_VECTORS, 64>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width);
break;
    */
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);

  if (*out != 1) {
    *out = 0;
  }
}

template <typename T>
void query_bp_stateful_contains_zero(const T *__restrict in, T *__restrict out,
                            const size_t count, const int32_t value_bit_width,
                            const int32_t unpack_n_values) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  constexpr auto UNPACK_N_VECTORS = 1;
  const auto n_vectors_per_block = 2 * UNPACK_N_VECTORS;
  const auto n_blocks = n_vecs / n_vectors_per_block;
  const auto n_threads = utils::get_n_lanes<T>() * 2;

  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(1);

	kernels::fls::global::bench::query_bp_stateful_contains_zero<T, UNPACK_N_VECTORS, 1>
			<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
																value_bit_width);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);

  if (*out != 1) {
    *out = 0;
  }
}

template <typename T>
void query_ffor_contains_zero(const T *__restrict in, T *__restrict out,
                              const size_t count, const int32_t value_bit_width,
                              const T *__restrict base_p,
                              const int32_t unpack_n_values) {
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_vectors_per_block = 2;
  const auto n_blocks = n_vecs / n_vectors_per_block;
  const auto n_threads = utils::get_n_lanes<T>() * n_vectors_per_block;

  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(1);
  GPUArray<T> device_base_p(1, base_p);

  switch (unpack_n_values) {
  default:
  case 1:
    kernels::fls::global::bench::query_ffor_contains_zero<
        T, 1, utils::get_values_per_lane<T>()>
        <<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                                  value_bit_width, device_base_p.get());
    break;
    /*
case 2:
kernels::fls::global::bench::query_ffor_contains_zero<
T, 2, utils::get_values_per_lane<T>()>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width, device_base_p.get());
break;
case 4:
kernels::fls::global::bench::query_ffor_contains_zero<
T, 4, utils::get_values_per_lane<T>()>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width, device_base_p.get());
break;
case 8:
kernels::fls::global::bench::query_ffor_contains_zero<
T, 8, utils::get_values_per_lane<T>()>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width, device_base_p.get());
break;
case 16:
kernels::fls::global::bench::query_ffor_contains_zero<
T, 16, utils::get_values_per_lane<T>()>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width, device_base_p.get());
break;
case 32:
kernels::fls::global::bench::query_ffor_contains_zero<
T, 32, utils::get_values_per_lane<T>()>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width, device_base_p.get());
break;
case 64:
kernels::fls::global::bench::query_ffor_contains_zero<
T, 64, utils::get_values_per_lane<T>()>
<<<n_blocks, n_threads>>>(device_in.get(), device_out.get(),
                      value_bit_width, device_base_p.get());
break;
    */
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);
}

} // namespace bench
} // namespace gpu
} // namespace fls
template void fls::gpu::bench::query_baseline_contains_zero<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count);
template void fls::gpu::bench::query_baseline_contains_zero<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out,
    const size_t count);
template void fls::gpu::bench::query_baseline_contains_zero<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out,
    const size_t count);
template void fls::gpu::bench::query_baseline_contains_zero<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out,
    const size_t count);
template void fls::gpu::bench::query_old_fls_contains_zero<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out,
    const size_t count, const int32_t value_bit_width);
template void fls::gpu::bench::query_bp_contains_zero<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);
template void fls::gpu::bench::query_bp_contains_zero<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);
template void fls::gpu::bench::query_bp_contains_zero<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);
template void fls::gpu::bench::query_bp_contains_zero<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);
template void fls::gpu::bench::query_bp_stateful_contains_zero<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);
template void fls::gpu::bench::query_bp_stateful_contains_zero<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);
template void fls::gpu::bench::query_bp_stateful_contains_zero<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);
template void fls::gpu::bench::query_bp_stateful_contains_zero<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);
template void fls::gpu::bench::query_ffor_contains_zero<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint8_t *__restrict base_p,
    const int32_t unpack_n_values);
template void fls::gpu::bench::query_ffor_contains_zero<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint16_t *__restrict base_p,
    const int32_t unpack_n_values);
template void fls::gpu::bench::query_ffor_contains_zero<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint32_t *__restrict base_p,
    const int32_t unpack_n_values);
template void fls::gpu::bench::query_ffor_contains_zero<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint64_t *__restrict base_p,
    const int32_t unpack_n_values);
