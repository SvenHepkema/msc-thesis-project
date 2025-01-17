#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "../alp/alp-bindings.hpp"

#ifndef KERNELS_HPP
#define KERNELS_HPP

namespace kernels {

enum KernelOption {
  CPU,
  TEST_STATELESS_1_1,
  TEST_STATEFUL_1_1,
  TEST_STATELESS_BRANCHLESS_1_1,
  TEST_STATEFUL_BRANCHLESS_1_1,
  QUERY_STATELESS_1_1,
};

struct KernelSpecification {
  const KernelOption spec;

  const unsigned n_vectors;
  const unsigned n_values;

  KernelSpecification() : spec(KernelOption::CPU), n_vectors(1), n_values(1) {}

  KernelSpecification(const KernelOption a_spec, const unsigned a_n_vectors,
                      const unsigned a_n_values)
      : spec(a_spec), n_vectors(a_n_vectors), n_values(a_n_values) {}
};

static inline const std::unordered_map<std::string, KernelSpecification>
    kernel_options{
        {"cpu", KernelSpecification(CPU, 1, 1)},
        {"test_stateless_1_1", KernelSpecification(TEST_STATELESS_1_1, 1, 1)},
        {"test_stateful_1_1", KernelSpecification(TEST_STATEFUL_1_1, 1, 1)},
        {"test_stateless_branchless_1_1",
         KernelSpecification(TEST_STATELESS_BRANCHLESS_1_1, 1, 1)},
        {"test_stateful_branchless_1_1",
         KernelSpecification(TEST_STATEFUL_BRANCHLESS_1_1, 1, 1)},
        {"query_stateless_1_1", KernelSpecification(QUERY_STATELESS_1_1, 1, 1)},
    };

namespace fls {
template <typename T>
void verify_bitunpack(const KernelSpecification spec, const T *__restrict in,
                      T *__restrict out, const size_t count,
                      const int32_t value_bit_width);
template <typename T>
void query_column_contains_zero(const KernelSpecification spec,
                                const T *__restrict in, T *__restrict out,
                                const size_t count,
                                const int32_t value_bit_width);
} // namespace fls
namespace gpualp {
template <typename T>
void verify_decompress_column(const KernelSpecification spec, T *__restrict out,
                              const alp::AlpCompressionData<T> *data);
template <typename T>
void query_column_contains_magic(const KernelSpecification spec,
                                 T *__restrict out,
                                 const alp::AlpCompressionData<T> *data,
                                 const T magic_value);

} // namespace gpualp

} // namespace kernels

extern template void kernels::fls::verify_bitunpack<uint8_t>(
    const kernels::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
extern template void kernels::fls::verify_bitunpack<uint16_t>(
    const kernels::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void kernels::fls::verify_bitunpack<uint64_t>(
    const kernels::KernelSpecification spec, const uint64_t *__restrict in,
    uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

extern template void kernels::fls::query_column_contains_zero<uint8_t>(
    const kernels::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
extern template void kernels::fls::query_column_contains_zero<uint16_t>(
    const kernels::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void kernels::fls::query_column_contains_zero<uint64_t>(
    const kernels::KernelSpecification spec, const uint64_t *__restrict in,
    uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

extern template void kernels::gpualp::verify_decompress_column<float>(
    const kernels::KernelSpecification spec, float *__restrict out,
    const alp::AlpCompressionData<float> *data);
extern template void kernels::gpualp::verify_decompress_column<double>(
    const kernels::KernelSpecification spec, double *__restrict out,
    const alp::AlpCompressionData<double> *data);

extern template void kernels::gpualp::query_column_contains_magic<float>(
    const ::kernels::KernelSpecification spec, float *__restrict out,
    const alp::AlpCompressionData<float> *data, const float magic_value);
extern template void kernels::gpualp::query_column_contains_magic<double>(
    const ::kernels::KernelSpecification spec, double *__restrict out,
    const alp::AlpCompressionData<double> *data, const double magic_value);

#endif // KERNELS_HPP
