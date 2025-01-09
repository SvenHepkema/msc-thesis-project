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
void contains_magic_stateless(T *__restrict out,
                              const alp::AlpCompressionData<T> *data,
                              const T magic_value);

template <typename T>
void contains_magic_stateful(T *__restrict out,
                             const alp::AlpCompressionData<T> *data,
                             const T magic_value);

template <typename T, unsigned UNPACK_N_VECTORS>
void contains_magic_stateful_extended(T *__restrict out,
                                      const alp::AlpCompressionData<T> *data,
                                      const T magic_value);

template <typename T>
void decode_multiple_alp_vectors(
    T *__restrict out, const std::vector<alp::AlpCompressionData<T> *> data);

} // namespace bench
} // namespace gpu
} // namespace alp

extern template void
alp::gpu::bench::decode_baseline<float>(float *__restrict out, const float *in,
                                        const size_t count);
extern template void alp::gpu::bench::decode_baseline<double>(
    double *__restrict out, const double *data, const size_t count);

extern template void alp::gpu::bench::contains_magic_stateless<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data,
    const float magic_value);
extern template void alp::gpu::bench::contains_magic_stateless<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data,
    const double magic_value);

extern template void alp::gpu::bench::contains_magic_stateful<float>(
    float *__restrict out, const alp::AlpCompressionData<float> *data,
    const float magic_value);
extern template void alp::gpu::bench::contains_magic_stateful<double>(
    double *__restrict out, const alp::AlpCompressionData<double> *data,
    const double magic_value);

extern template void
alp::gpu::bench::contains_magic_stateful_extended<float, 1>(
    float *__restrict out, const alp::AlpCompressionData<float> *data,
    const float magic_value);
extern template void
alp::gpu::bench::contains_magic_stateful_extended<double, 1>(
    double *__restrict out, const alp::AlpCompressionData<double> *data,
    const double magic_value);
extern template void
alp::gpu::bench::contains_magic_stateful_extended<float, 2>(
    float *__restrict out, const alp::AlpCompressionData<float> *data,
    const float magic_value);
extern template void
alp::gpu::bench::contains_magic_stateful_extended<double, 2>(
    double *__restrict out, const alp::AlpCompressionData<double> *data,
    const double magic_value);
extern template void
alp::gpu::bench::contains_magic_stateful_extended<float, 4>(
    float *__restrict out, const alp::AlpCompressionData<float> *data,
    const float magic_value);
extern template void
alp::gpu::bench::contains_magic_stateful_extended<double, 4>(
    double *__restrict out, const alp::AlpCompressionData<double> *data,
    const double magic_value);

extern template void alp::gpu::bench::decode_multiple_alp_vectors<float>(
    float *__restrict out,
    const std::vector<alp::AlpCompressionData<float> *> data);
extern template void alp::gpu::bench::decode_multiple_alp_vectors<double>(
    double *__restrict out,
    const std::vector<alp::AlpCompressionData<double> *> data);

#endif // ALP_BENCHMARK_KERNELS_BINDINGS_HPP
