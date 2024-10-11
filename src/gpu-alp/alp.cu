#include "alp.cuh"

namespace constant_memory {

__host__ void load_alp_constants() {
  cudaMemcpyToSymbol(F_FACT_ARRAY,
                     alp::Constants<float>::FACT_ARR,
                     F_FACT_ARR_COUNT * sizeof(int32_t));
  cudaMemcpyToSymbol(F_FRAC_ARRAY,
                     alp::Constants<float>::FRAC_ARR,
                     F_FRAC_ARR_COUNT * sizeof(float));

  cudaMemcpyToSymbol(D_FACT_ARRAY,
                     alp::Constants<double>::FACT_ARR,
                     D_FACT_ARR_COUNT * sizeof(int64_t));
  cudaMemcpyToSymbol(D_FRAC_ARRAY,
                     alp::Constants<double>::FRAC_ARR,
                     D_FRAC_ARR_COUNT * sizeof(double));
}

}
