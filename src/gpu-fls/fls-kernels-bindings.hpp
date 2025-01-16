#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#ifndef KERNELS_HPP
#define KERNELS_HPP

namespace kernels {

enum KernelOption {
  CPU,
  TEST_STATELESS_1_1,
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
        {"query_stateless_1_1", KernelSpecification(QUERY_STATELESS_1_1, 1, 1)},
    };

template <typename T>
void verify_bitunpack(const KernelSpecification spec, const T *__restrict in,
                      T *__restrict out, const size_t count,
                      const int32_t value_bit_width);
template <typename T>
void query_column_contains_zero(const KernelSpecification spec,
                                const T *__restrict in, T *__restrict out,
                                const size_t count,
                                const int32_t value_bit_width);

} // namespace kernels

extern template void kernels::verify_bitunpack<uint8_t>(
    const kernels::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
extern template void kernels::verify_bitunpack<uint16_t>(
    const kernels::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void kernels::verify_bitunpack<uint64_t>(
    const kernels::KernelSpecification spec, const uint64_t *__restrict in,
    uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void kernels::query_column_contains_zero<uint8_t>(
    const kernels::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
extern template void kernels::query_column_contains_zero<uint16_t>(
    const kernels::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void kernels::query_column_contains_zero<uint64_t>(
    const kernels::KernelSpecification spec, const uint64_t *__restrict in,
    uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

#endif // KERNELS_HPP
