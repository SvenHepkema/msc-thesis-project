#include <cstddef>
#include <cstdint>

#ifndef FLS_BENCHMARK_KERNELS_BINDING_H
#define FLS_BENCHMARK_KERNELS_BINDING_H

namespace fls {
namespace gpu {
namespace bench {

template <typename T>
void query_baseline_contains_zero(const T *__restrict in, T *__restrict out,
                                  const size_t count);

template <typename T>
void query_bp_contains_zero(const T *__restrict in, T *__restrict out,
                            const size_t count, const int32_t value_bit_width,
                            const int32_t unpack_n_values);

template <typename T>
void query_ffor_contains_zero(const T *__restrict in, T *__restrict out,
                              const size_t count, const int32_t value_bit_width,
                              const T *__restrict base_p,
                              const int32_t unpack_n_values);

} // namespace bench
} // namespace gpu
} // namespace fls

extern template void fls::gpu::bench::query_baseline_contains_zero<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count);
extern template void fls::gpu::bench::query_baseline_contains_zero<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out,
    const size_t count);
extern template void fls::gpu::bench::query_baseline_contains_zero<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out,
    const size_t count);
extern template void fls::gpu::bench::query_baseline_contains_zero<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out,
    const size_t count);
extern template void fls::gpu::bench::query_bp_contains_zero<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);
extern template void fls::gpu::bench::query_bp_contains_zero<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);
extern template void fls::gpu::bench::query_bp_contains_zero<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);
extern template void fls::gpu::bench::query_bp_contains_zero<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const int32_t unpack_n_values);

extern template void fls::gpu::bench::query_ffor_contains_zero<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint8_t *__restrict base_p,
    const int32_t unpack_n_values);
extern template void fls::gpu::bench::query_ffor_contains_zero<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint16_t *__restrict base_p,
    const int32_t unpack_n_values);
extern template void fls::gpu::bench::query_ffor_contains_zero<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint32_t *__restrict base_p,
    const int32_t unpack_n_values);
extern template void fls::gpu::bench::query_ffor_contains_zero<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint64_t *__restrict base_p,
    const int32_t unpack_n_values);

#endif // FLS_BENCHMARK_KERNELS_BINDING_H
