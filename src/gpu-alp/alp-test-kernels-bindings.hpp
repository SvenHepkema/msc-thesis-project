#include <cstddef>
#include <cstdint>

#include "../alp/alp-bindings.hpp"

#ifndef ALP_TEST_KERNELS_BINDINGS_HPP
#define ALP_TEST_KERNELS_BINDINGS_HPP

namespace alp {
namespace gpu {
namespace test {

template <typename T>
void decode_alp_vector_stateless(T *__restrict out,
                                const alp::AlpCompressionData<T> *data);
template <typename T>
void decode_alp_vector_stateful(T *__restrict out,
                                const alp::AlpCompressionData<T> *data);

template <typename T, unsigned UNPACK_N_VECTORS = 1>
void decode_alp_vector_stateful_extended(T *__restrict out,
                                const alp::AlpCompressionData<T> *data);

template <typename T>
void decode_complete_alprd_vector(T *__restrict out,
                                  const alp::AlpRdCompressionData<T> *data);

} // namespace test
} // namespace gpu
} // namespace alp

extern template void alp::gpu::test::decode_alp_vector_stateless<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::test::decode_alp_vector_stateless<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

extern template void alp::gpu::test::decode_alp_vector_stateful<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::test::decode_alp_vector_stateful<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

extern template void alp::gpu::test::decode_alp_vector_stateful_extended<float, 1>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::test::decode_alp_vector_stateful_extended<double, 1>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
extern template void alp::gpu::test::decode_alp_vector_stateful_extended<float, 2>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::test::decode_alp_vector_stateful_extended<double, 2>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);
extern template void alp::gpu::test::decode_alp_vector_stateful_extended<float, 4>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::test::decode_alp_vector_stateful_extended<double, 4>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

extern template void alp::gpu::test::decode_complete_alprd_vector<float>(
    float *__restrict out, const alp::AlpRdCompressionData<float> *data);
extern template void alp::gpu::test::decode_complete_alprd_vector<double>(
    double *__restrict out, const alp::AlpRdCompressionData<double> *data);
#endif // ALP_TEST_KERNELS_BINDINGS_HPP
