#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "../alp/alp-bindings.hpp"
#include "../common/runspec.hpp"

#ifndef KERNELS_HPP
#define KERNELS_HPP

namespace kernels {

namespace fls {
template <typename T>
void verify_bitunpack(const runspec::KernelSpecification spec,
                      const T *__restrict in, T *__restrict out,
                      const size_t count, const int32_t value_bit_width);
template <typename T>
void query_column_contains_zero(const runspec::KernelSpecification spec,
                                const T *__restrict in, T *__restrict out,
                                const size_t count,
                                const int32_t value_bit_width);
} // namespace fls
namespace gpualp {
template <typename T>
void verify_decompress_column(const runspec::KernelSpecification spec,
                              T *__restrict out,
                              const alp::AlpCompressionData<T> *data);
template <typename T>
void query_column_contains_magic(const runspec::KernelSpecification spec,
                                 T *__restrict out,
                                 const alp::AlpCompressionData<T> *data,
                                 const T magic_value);

} // namespace gpualp

} // namespace kernels

extern template void kernels::fls::verify_bitunpack<uint8_t>(
    const runspec::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
extern template void kernels::fls::verify_bitunpack<uint16_t>(
    const runspec::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void kernels::fls::verify_bitunpack<uint64_t>(
    const runspec::KernelSpecification spec, const uint64_t *__restrict in,
    uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

extern template void kernels::fls::query_column_contains_zero<uint8_t>(
    const runspec::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
extern template void kernels::fls::query_column_contains_zero<uint16_t>(
    const runspec::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void kernels::fls::query_column_contains_zero<uint64_t>(
    const runspec::KernelSpecification spec, const uint64_t *__restrict in,
    uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

extern template void kernels::gpualp::verify_decompress_column<float>(
    const runspec::KernelSpecification spec, float *__restrict out,
    const alp::AlpCompressionData<float> *data);
extern template void kernels::gpualp::verify_decompress_column<double>(
    const runspec::KernelSpecification spec, double *__restrict out,
    const alp::AlpCompressionData<double> *data);

extern template void kernels::gpualp::query_column_contains_magic<float>(
    const runspec::KernelSpecification spec, float *__restrict out,
    const alp::AlpCompressionData<float> *data, const float magic_value);
extern template void kernels::gpualp::query_column_contains_magic<double>(
    const runspec::KernelSpecification spec, double *__restrict out,
    const alp::AlpCompressionData<double> *data, const double magic_value);

#endif // KERNELS_HPP
