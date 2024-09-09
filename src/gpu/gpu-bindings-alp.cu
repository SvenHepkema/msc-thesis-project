#include <cstddef>
#include <cstdint>

#include "../consts.hpp"
#include "alp.cuh"
#include "alp-global.cuh"
#include "gpu-bindings-alp.hpp"
#include "gpu-utils.cuh"


namespace gpu {

template <typename T>
void alp(T *__restrict out, const alp::AlpCompressionData<T> *data) {
  using UINT_T =
      typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;
  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  const auto n_blocks = n_vecs;

  GPUArray<T> d_out(count);
  GPUArray<UINT_T> d_ffor_array(count, data->ffor_array);

  GPUArray<UINT_T> d_ffor_bases(n_vecs, data->ffor_bases);
  GPUArray<uint8_t> d_bit_widths(n_vecs, data->bit_widths);
  GPUArray<uint8_t> d_exponents(n_vecs, data->exponents);
  GPUArray<uint8_t> d_factors(n_vecs, data->factors);

  AlpColumn<T> alp_data = {d_ffor_array.get(), d_ffor_bases.get(),
                         d_bit_widths.get(), d_exponents.get(),
                         d_factors.get()};
	constant_memory::load_alp_constants();

  alp_global<T, UINT_T, 1, utils::get_values_per_lane<T>()>
      <<<n_blocks, utils::get_n_lanes<T>()>>>(d_out.get(), alp_data);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);
}

} // namespace gpu

template void gpu::alp<double>(double *__restrict out,
                               const alp::AlpCompressionData<double> *data);
