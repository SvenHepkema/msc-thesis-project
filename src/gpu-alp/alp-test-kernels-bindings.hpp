#include <cstddef>
#include <cstdint>

#include "../alp/alp-bindings.hpp"

#ifndef ALP_TEST_KERNELS_BINDINGS_HPP
#define ALP_TEST_KERNELS_BINDINGS_HPP

namespace alp {
namespace gpu {
namespace test {

template <typename T>
void decode_complete_alp_vector(T *__restrict out,
                                const alp::AlpCompressionData<T> *data);
template <typename T>
void decode_alp_vector_into_lane(T *__restrict out,
                                const alp::AlpCompressionData<T> *data);
template <typename T>
void decode_alp_vector_with_state(T *__restrict out,
                                const alp::AlpCompressionData<T> *data);

template <typename T>
void decode_alp_vector_with_extended_state(T *__restrict out,
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

extern template void alp::gpu::test::decode_alp_vector_into_lane<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::test::decode_alp_vector_into_lane<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

extern template void alp::gpu::test::decode_alp_vector_with_state<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::test::decode_alp_vector_with_state<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

extern template void alp::gpu::test::decode_alp_vector_with_extended_state<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::test::decode_alp_vector_with_extended_state<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

extern template void alp::gpu::test::decode_complete_alprd_vector<float>(
    float *__restrict out, const alp::AlpRdCompressionData<float> *data);
extern template void alp::gpu::test::decode_complete_alprd_vector<double>(
    double *__restrict out, const alp::AlpRdCompressionData<double> *data);
#endif // ALP_TEST_KERNELS_BINDINGS_HPP
