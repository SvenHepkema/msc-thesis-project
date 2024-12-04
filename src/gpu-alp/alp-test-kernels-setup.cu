#include <cstddef>
#include <cstdint>

#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "../gpu-common/gpu-utils.cuh"
#include "alp-test-kernels-bindings.hpp"
#include "alp-test-kernels-global.cuh"
#include "alp.cuh"
#include "src/alp/config.hpp"
#include "alp-utils.cuh"

namespace alp {
namespace gpu {
namespace test {

template <typename T>
void decode_complete_alp_vector(T *__restrict out,
                                const alp::AlpCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_blocks = n_vecs;

  GPUArray<T> d_out(count);
  AlpColumn<T> gpu_alp_column = transfer::copy_alp_column_to_gpu(data);
  constant_memory::load_alp_constants<T>();

  kernels::global::test::decode_complete_alp_vector<
      T, UINT_T, 1, utils::get_values_per_lane<T>()>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(d_out.get(), gpu_alp_column);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);
	transfer::destroy_alp_column(gpu_alp_column);
}

template <typename T>
void decode_alp_vector_into_lane(T *__restrict out,
                                const alp::AlpCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_blocks = n_vecs;

  GPUArray<T> d_out(count);
  AlpColumn<T> gpu_alp_column = transfer::copy_alp_column_to_gpu(data);
  constant_memory::load_alp_constants<T>();

  kernels::global::test::decode_alp_vector_into_lane<
      T, UINT_T, 1, 1>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(d_out.get(), gpu_alp_column);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);
	transfer::destroy_alp_column(gpu_alp_column);
}

template <typename T>
void decode_alp_vector_with_state(T *__restrict out,
                                const alp::AlpCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_blocks = n_vecs;

  GPUArray<T> d_out(count);
  AlpColumn<T> gpu_alp_column = transfer::copy_alp_column_to_gpu(data);
  constant_memory::load_alp_constants<T>();

  kernels::global::test::decode_alp_vector_with_state<
      T, UINT_T, 1, 1>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(d_out.get(), gpu_alp_column);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);
	transfer::destroy_alp_column(gpu_alp_column);
}

template <typename T>
void decode_alp_vector_with_extended_state(T *__restrict out,
                                const alp::AlpCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_blocks = n_vecs;

  GPUArray<T> d_out(count);
  auto gpu_alp_column = transfer::copy_alp_extended_column_to_gpu(data);
  constant_memory::load_alp_constants<T>();

  kernels::global::test::decode_alp_vector_with_extended_state<
      T, UINT_T, 1, 1>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(d_out.get(), gpu_alp_column);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);
	transfer::destroy_alp_column(gpu_alp_column);
}

template <typename T>
void decode_complete_alprd_vector(T *__restrict out,
                                  const alp::AlpRdCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_blocks = n_vecs;

  GPUArray<T> d_out(count);

  GPUArray<uint16_t> d_left_ffor_array(count, data->left_ffor.array);
  GPUArray<uint16_t> d_left_ffor_bases(n_vecs, data->left_ffor.bases);
  GPUArray<uint8_t> d_left_bit_widths(n_vecs, data->left_ffor.bit_widths);

  GPUArray<UINT_T> d_right_ffor_array(count, data->right_ffor.array);
  GPUArray<UINT_T> d_right_ffor_bases(n_vecs, data->right_ffor.bases);
  GPUArray<uint8_t> d_right_bit_widths(n_vecs, data->right_ffor.bit_widths);

  GPUArray<uint16_t> d_left_parts_dicts(
      n_vecs * alp::config::MAX_RD_DICTIONARY_SIZE, data->left_parts_dicts);

  GPUArray<uint16_t> d_exceptions(count, data->exceptions.exceptions);
  GPUArray<uint16_t> d_exception_positions(count, data->exceptions.positions);
  GPUArray<uint16_t> d_exception_counts(n_vecs, data->exceptions.counts);

  AlpRdColumn<T> alp_data = {
      d_left_ffor_array.get(),     d_left_ffor_bases.get(),
      d_left_bit_widths.get(),     d_right_ffor_array.get(),
      d_right_ffor_bases.get(),    d_right_bit_widths.get(),
      d_left_parts_dicts.get(),    d_exceptions.get(),
      d_exception_positions.get(), d_exception_counts.get(),
  };
  constant_memory::load_alp_constants<T>();

  kernels::global::test::decode_complete_alprd_vector<
      T, UINT_T, 1, utils::get_values_per_lane<T>()>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(d_out.get(), alp_data);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);
}

} // namespace test
} // namespace gpu
} // namespace alp

template void alp::gpu::test::decode_complete_alp_vector<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::test::decode_complete_alp_vector<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
template void alp::gpu::test::decode_alp_vector_into_lane<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::test::decode_alp_vector_into_lane<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
template void alp::gpu::test::decode_alp_vector_with_state<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::test::decode_alp_vector_with_state<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
template void alp::gpu::test::decode_alp_vector_with_extended_state<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::test::decode_alp_vector_with_extended_state<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
template void alp::gpu::test::decode_complete_alprd_vector<float>(
    float *__restrict out, const alp::AlpRdCompressionData<float> *data);
template void alp::gpu::test::decode_complete_alprd_vector<double>(
    double *__restrict out, const alp::AlpRdCompressionData<double> *data);
