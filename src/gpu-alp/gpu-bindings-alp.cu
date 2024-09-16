#include <cstddef>
#include <cstdint>

#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "alp-global.cuh"
#include "alp.cuh"
#include "gpu-bindings-alp.hpp"
#include "../gpu-common/gpu-utils.cuh"

namespace gpu {

template <typename T>
void alp(T *__restrict out, const alp::AlpCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_blocks = n_vecs;

  GPUArray<T> d_out(count);
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
  constant_memory::load_alp_constants();

  alp_global<T, UINT_T, 1, utils::get_values_per_lane<T>()>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(d_out.get(), alp_data);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);
}

} // namespace gpu

template void gpu::alp<float>(float *__restrict out,
                               const alp::AlpCompressionData<float> *data);
template void gpu::alp<double>(double *__restrict out,
                               const alp::AlpCompressionData<double> *data);
