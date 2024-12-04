#include <cstddef>
#include <cstdint>

#include "../alp/alp-bindings.hpp"

#ifndef ALP_BENCHMARK_KERNELS_BINDINGS_HPP
#define ALP_BENCHMARK_KERNELS_BINDINGS_HPP

namespace alp {
namespace gpu {
namespace bench {

template <typename T>
void decode_baseline(T *__restrict out, const T *in, const size_t count);

template <typename T>
void decode_complete_alp_vector(T *__restrict out,
                                const alp::AlpCompressionData<T> *data);

template <typename T>
void decode_alp_vector_with_state(T *__restrict out,
                                const alp::AlpCompressionData<T> *data);

template <typename T>
void decode_alp_vector_with_extended_state(T *__restrict out,
                                const alp::AlpCompressionData<T> *data);

template <typename T>
void decode_multiple_alp_vectors(
    T *__restrict out, const std::vector<alp::AlpCompressionData<T> *> data);

template <typename T>
void decode_complete_alprd_vector(T *__restrict out,
                                  const alp::AlpRdCompressionData<T> *data);

} // namespace bench
} // namespace gpu
} // namespace alp

extern template void
alp::gpu::bench::decode_baseline<float>(float *__restrict out, const float *in,
                                        const size_t count);
extern template void alp::gpu::bench::decode_baseline<double>(
    double *__restrict out, const double *data, const size_t count);

extern template void alp::gpu::bench::decode_complete_alp_vector<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::bench::decode_complete_alp_vector<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

extern template void alp::gpu::bench::decode_alp_vector_with_state<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::bench::decode_alp_vector_with_state<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

extern template void alp::gpu::bench::decode_alp_vector_with_extended_state<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data);
extern template void alp::gpu::bench::decode_alp_vector_with_extended_state<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data);

extern template void alp::gpu::bench::decode_multiple_alp_vectors<float>(
    float *__restrict out,
    const std::vector<alp::AlpCompressionData<float> *> data);
extern template void alp::gpu::bench::decode_multiple_alp_vectors<double>(
    double *__restrict out,
    const std::vector<alp::AlpCompressionData<double> *> data);

extern template void alp::gpu::bench::decode_complete_alprd_vector<float>(
    float *__restrict out, const alp::AlpRdCompressionData<float> *data);
extern template void alp::gpu::bench::decode_complete_alprd_vector<double>(
    double *__restrict out, const alp::AlpRdCompressionData<double> *data);
#endif // ALP_BENCHMARK_KERNELS_BINDINGS_HPP
