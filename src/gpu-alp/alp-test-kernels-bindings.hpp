#include <cstddef>
#include <cstdint>

#include "../alp/alp-bindings.hpp"

#ifndef GPU_ALP_HPP
#define GPU_ALP_HPP

namespace gpu {
template <typename T>
void test_alp_complete_vector_decoding(T *__restrict out, const alp::AlpCompressionData<T> *data);

template <typename T>
void test_alprd_complete_vector_decoding(T *__restrict out, const alp::AlpRdCompressionData<T> *data);
} // namespace gpu

extern template void gpu::test_alp_complete_vector_decoding<float>(float *__restrict out,
                                      const alp::AlpCompressionData<float> *data);
extern template void gpu::test_alp_complete_vector_decoding<double>(double *__restrict out,
                                      const alp::AlpCompressionData<double> *data);

extern template void gpu::test_alprd_complete_vector_decoding<float>(float *__restrict out,
                                      const alp::AlpRdCompressionData<float> *data);
extern template void gpu::test_alprd_complete_vector_decoding<double>(double *__restrict out,
                                      const alp::AlpRdCompressionData<double> *data);
#endif // GPU_ALP_HPP
