#include <cstddef>
#include <cstdint>

#include "../alp/alp-bindings.hpp"

#ifndef GPU_ALP_HPP
#define GPU_ALP_HPP

namespace alp {
namespace gpu {
namespace test {

template <typename T>
void decode_complete_alp_vector(T *__restrict out,
                                const alp::AlpCompressionData<T> *data);

template <typename T>
void decode_complete_alprd_vector(T *__restrict out,
                                  const alp::AlpRdCompressionData<T> *data);

} // namespace test
} // namespace gpu
} // namespace alp

extern template void alp::gpu::test::decode_complete_alp_vector<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::test::decode_complete_alp_vector<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

extern template void alp::gpu::test::decode_complete_alprd_vector<float>(
    float *__restrict out, const alp::AlpRdCompressionData<float> *data);
extern template void alp::gpu::test::decode_complete_alprd_vector<double>(
    double *__restrict out, const alp::AlpRdCompressionData<double> *data);
#endif // GPU_ALP_HPP
