#include <cstddef>
#include <cstdint>

#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "../gpu-common/gpu-utils.cuh"
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

  kernels::global::bench::decode_baseline<T, T, 1, 1>
      <<<n_blocks, n_threads>>>(d_out.get(), d_in.get());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);

	if (*out != static_cast<T>(true)) {
		*out = static_cast<T>(false);
	}
}

template <typename T>
void decode_complete_alp_vector(T *__restrict out,
                                const alp::AlpCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_warps_per_block = 2;
  const auto n_blocks = n_vecs / n_warps_per_block;
  const auto n_threads = n_warps_per_block * consts::THREADS_PER_WARP;

  GPUArray<T> d_out(1);
  GPUArray<UINT_T> d_ffor_array(count, data->ffor.array);

  GPUArray<UINT_T> d_ffor_bases(n_vecs, data->ffor.bases);
  GPUArray<uint8_t> d_bit_widths(n_vecs, data->ffor.bit_widths);
  GPUArray<uint8_t> d_exponents(n_vecs, data->exponents);
  GPUArray<uint8_t> d_factors(n_vecs, data->factors);

  GPUArray<T> d_exceptions(count, data->exceptions.exceptions);
  GPUArray<uint16_t> d_exception_positions(count, data->exceptions.positions);
  GPUArray<uint16_t> d_exception_counts(n_vecs, data->exceptions.counts);

  AlpColumn<T> alp_data = {
      d_ffor_array.get(),          d_ffor_bases.get(),      d_bit_widths.get(),
      d_exponents.get(),           d_factors.get(),         d_exceptions.get(),
      d_exception_positions.get(), d_exception_counts.get()};
  constant_memory::load_alp_constants<T>();

  kernels::global::bench::decode_complete_alp_vector<
      T, UINT_T, 1, 4>
      <<<n_blocks, n_threads>>>(d_out.get(), alp_data);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);

	if (*out != static_cast<T>(true)) {
		*out = static_cast<T>(false);
	}
}

template <typename T>
void decode_complete_alprd_vector(T *__restrict out,
                                  const alp::AlpRdCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_blocks = n_vecs;

  GPUArray<T> d_out(1);

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

  kernels::global::bench::decode_complete_alprd_vector<
      T, UINT_T, 1, utils::get_values_per_lane<T>()>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(d_out.get(), alp_data);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);

	if (*out != static_cast<T>(true)) {
		*out = static_cast<T>(false);
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
template void alp::gpu::bench::decode_complete_alp_vector<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
template void alp::gpu::bench::decode_complete_alp_vector<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
template void alp::gpu::bench::decode_complete_alprd_vector<float>(
    float *__restrict out, const alp::AlpRdCompressionData<float> *data);
template void alp::gpu::bench::decode_complete_alprd_vector<double>(
    double *__restrict out, const alp::AlpRdCompressionData<double> *data);
