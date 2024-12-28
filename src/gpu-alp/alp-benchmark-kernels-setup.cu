#include <cstddef>
#include <cstdint>
#include <tuple>

#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "../gpu-common/gpu-utils.cuh"
#include "alp-utils.cuh"
#include "alp-benchmark-kernels-bindings.hpp"
#include "alp-benchmark-kernels-global.cuh"
#include "alp.cuh"
#include "src/alp/config.hpp"


namespace alp {
namespace gpu {
namespace bench {

template <typename T>
void decode_baseline(T *__restrict out, const T *in, const size_t count) {
  GPUArray<T> d_in(count, in);
  GPUArray<T> d_out(1);

  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_warps_per_block = 2;
  const auto n_blocks = n_vecs / n_warps_per_block;
  const auto n_threads = n_warps_per_block * consts::THREADS_PER_WARP;

  kernels::global::bench::decode_baseline<T, T, 1, 1><<<n_blocks, n_threads>>>(
      d_out.get(), d_in.get(), utils::get_n_lanes<T>());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);

  if (*out != static_cast<T>(true)) {
    *out = static_cast<T>(false);
  }
}

template <typename T>
void contains_magic_stateless(T *__restrict out,
                                const alp::AlpCompressionData<T> *data) {
  constexpr int32_t UNPACK_N_VECTORS = 1;
  constexpr int32_t UNPACK_N_VALUES = 1;
  using UINT_T = typename utils::same_width_uint<T>::type;

  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_warps_per_block = 2;
  const auto n_blocks = n_vecs / n_warps_per_block;
  const auto n_threads = n_warps_per_block * consts::THREADS_PER_WARP;

  GPUArray<T> d_out(1);
  AlpColumn<T> gpu_alp_column = transfer::copy_alp_column_to_gpu(data);
  constant_memory::load_alp_constants<T>();

  kernels::global::bench::contains_magic_stateless<
      T, UINT_T, UNPACK_N_VECTORS, UNPACK_N_VALUES>
      <<<n_blocks, n_threads>>>(d_out.get(), gpu_alp_column);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);

  if (*out != static_cast<T>(true)) {
    *out = static_cast<T>(false);
  }

	transfer::destroy_alp_column(gpu_alp_column);
}

template <typename T>
void contains_magic_stateful(T *__restrict out,
                                  const alp::AlpCompressionData<T> *data) {
  constexpr int32_t UNPACK_N_VECTORS = 1;
  constexpr int32_t UNPACK_N_VALUES = 1;
  using UINT_T = typename utils::same_width_uint<T>::type;

  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_warps_per_block = 2;
  const auto n_blocks = n_vecs / n_warps_per_block;
  const auto n_threads = n_warps_per_block * consts::THREADS_PER_WARP;

  GPUArray<T> d_out(1);
  AlpColumn<T> gpu_alp_column = transfer::copy_alp_column_to_gpu(data);
  constant_memory::load_alp_constants<T>();

  kernels::global::bench::contains_magic_stateful<
      T, UINT_T, UNPACK_N_VECTORS, UNPACK_N_VALUES>
      <<<n_blocks, n_threads>>>(d_out.get(), gpu_alp_column);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);

  if (*out != static_cast<T>(true)) {
    *out = static_cast<T>(false);
  }

	transfer::destroy_alp_column(gpu_alp_column);
}

template <typename T, unsigned UNPACK_N_VECTORS = 1>
void contains_magic_stateful_extended(T *__restrict out,
                                  const alp::AlpCompressionData<T> *data) {
  constexpr int32_t UNPACK_N_VALUES = 1;
  using UINT_T = typename utils::same_width_uint<T>::type;

  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_warps_per_block = 2;
  const auto n_blocks = n_vecs / (n_warps_per_block * UNPACK_N_VECTORS);
  const auto n_threads = n_warps_per_block * consts::THREADS_PER_WARP;

  GPUArray<T> d_out(1);
  auto gpu_alp_column = transfer::copy_alp_extended_column_to_gpu(data);
  constant_memory::load_alp_constants<T>();

  kernels::global::bench::contains_magic_stateful_extended<
      T, UINT_T, UNPACK_N_VECTORS, UNPACK_N_VALUES>
      <<<n_blocks, n_threads>>>(d_out.get(), gpu_alp_column);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);

  if (*out != static_cast<T>(true)) {
    *out = static_cast<T>(false);
  }

	transfer::destroy_alp_column(gpu_alp_column);
}


template <typename T>
void decode_multiple_alp_vectors(
    T *__restrict out, const std::vector<alp::AlpCompressionData<T> *> data) {
  using UINT_T = typename utils::same_width_uint<T>::type;

  constexpr int32_t UNPACK_N_VECTORS = 1;
  constexpr int32_t UNPACK_N_VALUES = 1;

  const auto count = data[0]->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_warps_per_block = 2;
  const auto n_blocks = n_vecs / n_warps_per_block;
  const auto n_threads = n_warps_per_block * consts::THREADS_PER_WARP;
  constant_memory::load_alp_constants<T>();

  GPUArray<T> d_out(1);
  std::vector<AlpColumn<T>> gpu_alp_columns(0);
  for (auto column : data) {
    gpu_alp_columns.push_back(transfer::copy_alp_column_to_gpu(column));
  }
  switch (gpu_alp_columns.size()) {
  default:
  case 2:
    kernels::global::bench::decode_multiple_alp_vectors<
        T, UINT_T, UNPACK_N_VECTORS, UNPACK_N_VALUES><<<n_blocks, n_threads>>>(
        d_out.get(), gpu_alp_columns[0], gpu_alp_columns[1]);
    break;
  case 3:
    kernels::global::bench::decode_multiple_alp_vectors<
        T, UINT_T, UNPACK_N_VECTORS, UNPACK_N_VALUES>
        <<<n_blocks, n_threads>>>(d_out.get(), gpu_alp_columns[0],
                                  gpu_alp_columns[1], gpu_alp_columns[2]);
    break;
  case 4:
    kernels::global::bench::decode_multiple_alp_vectors<
        T, UINT_T, UNPACK_N_VECTORS, UNPACK_N_VALUES><<<n_blocks, n_threads>>>(
        d_out.get(), gpu_alp_columns[0], gpu_alp_columns[1], gpu_alp_columns[2],
        gpu_alp_columns[3]);
    break;
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);

  if (*out != static_cast<T>(true)) {
    *out = static_cast<T>(false);
  }

  for (auto column : gpu_alp_columns) {
		transfer::destroy_alp_column(column);
  }
}

} // namespace bench
} // namespace gpu
} // namespace alp

template void alp::gpu::bench::decode_baseline<float>(float *__restrict out,
                                                      const float *in,
                                                      const size_t count);
template void alp::gpu::bench::decode_baseline<double>(double *__restrict out,
                                                       const double *in,
                                                       const size_t count);

template void alp::gpu::bench::contains_magic_stateless<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::bench::contains_magic_stateless<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

template void alp::gpu::bench::contains_magic_stateful<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::bench::contains_magic_stateful<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

template void alp::gpu::bench::contains_magic_stateful_extended<float, 1>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::bench::contains_magic_stateful_extended<double, 1>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
template void alp::gpu::bench::contains_magic_stateful_extended<float, 2>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::bench::contains_magic_stateful_extended<double, 2>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
template void alp::gpu::bench::contains_magic_stateful_extended<float, 4>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::bench::contains_magic_stateful_extended<double, 4>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

template void alp::gpu::bench::decode_multiple_alp_vectors<float>(
    float *__restrict out,
    const std::vector<alp::AlpCompressionData<float> *> data);
template void alp::gpu::bench::decode_multiple_alp_vectors<double>(
    double *__restrict out,
    const std::vector<alp::AlpCompressionData<double> *> data);
