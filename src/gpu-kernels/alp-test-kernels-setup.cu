#include <cstddef>
#include <cstdint>

#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "gpu-utils.cuh"
#include "alp-test-kernels-bindings.hpp"
#include "alp-test-kernels-global.cuh"
#include "alp-utils.cuh"
#include "alp.cuh"
#include "src/alp/config.hpp"

namespace alp {
namespace gpu {
namespace test {

template <typename T>
void decode_alp_vector_stateless(T *__restrict out,
                                 const alp::AlpCompressionData<T> *data) {
  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_blocks = n_vecs;

  GPUArray<T> d_out(count);
  AlpColumn<T> gpu_alp_column = transfer::copy_alp_column_to_gpu(data);
  constant_memory::load_alp_constants<T>();

  kernels::global::test::decode_alp_vector_stateless<T, 1, 1>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(d_out.get(), gpu_alp_column);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);
  transfer::destroy_alp_column(gpu_alp_column);
}

template <typename T>
void decode_alp_vector_stateful(T *__restrict out,
                                const alp::AlpCompressionData<T> *data) {
  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_blocks = n_vecs;

  GPUArray<T> d_out(count);
  AlpColumn<T> gpu_alp_column = transfer::copy_alp_column_to_gpu(data);
  constant_memory::load_alp_constants<T>();

  kernels::global::test::decode_alp_vector_stateful<T, 1, 1>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(d_out.get(), gpu_alp_column);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);
  transfer::destroy_alp_column(gpu_alp_column);
}

template <typename T, unsigned UNPACK_N_VECTORS>
void decode_alp_vector_stateful_extended(
    T *__restrict out, const alp::AlpCompressionData<T> *data) {
  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_blocks = n_vecs / UNPACK_N_VECTORS;
  const auto n_threads = utils::get_n_lanes<T>();

  GPUArray<T> d_out(count);
  auto gpu_alp_column = transfer::copy_alp_extended_column_to_gpu(data);
  constant_memory::load_alp_constants<T>();

  kernels::global::test::decode_alp_vector_stateful_extended<
      T, UNPACK_N_VECTORS, 1>
      <<<n_blocks, n_threads>>>(d_out.get(), gpu_alp_column);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);
  transfer::destroy_alp_column(gpu_alp_column);
}

} // namespace test
} // namespace gpu
} // namespace alp

template void alp::gpu::test::decode_alp_vector_stateless<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::test::decode_alp_vector_stateless<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
template void alp::gpu::test::decode_alp_vector_stateful<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::test::decode_alp_vector_stateful<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
template void alp::gpu::test::decode_alp_vector_stateful_extended<float, 1>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::test::decode_alp_vector_stateful_extended<double, 1>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
template void alp::gpu::test::decode_alp_vector_stateful_extended<float, 2>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::test::decode_alp_vector_stateful_extended<double, 2>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
template void alp::gpu::test::decode_alp_vector_stateful_extended<float, 4>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::test::decode_alp_vector_stateful_extended<double, 4>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
