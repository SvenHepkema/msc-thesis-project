#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../engine/datageneration.hpp"
#include "../engine/verification.hpp"

#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "../gpu-kernels/kernels-bindings.hpp"

#include "../engine/queries.h"

namespace verify_benchmarkers {

	/*
template <typename T>
verification::VerificationResult<T>
verify_bench_float_baseline(const size_t a_count,
                            [[maybe_unused]] const std::string dataset_name) {
  auto value_bit_widths = verification::generate_integer_range<int32_t>(-1);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(),
          queries::baseline::cpu::FloatAnyValueIsMagicFn<T>(),
          queries::baseline::gpu::FloatAnyValueIsMagicFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
verify_bench_alp_stateless(const size_t a_count,
                           [[maybe_unused]] const std::string dataset_name) {
  auto value_bit_widths = verification::generate_integer_range<int32_t>(0, 1000);
  auto [data, generator] =
      data::lambda::get_reusable_compressed_binary_column<T>(dataset_name,
                                                             a_count);

  auto result = verification::run_verifier_on_parameters<
      T, alp::ALPMagicCompressionData<T>, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<
          T, alp::ALPMagicCompressionData<T>, int32_t, int32_t>(
          generator, queries::ALP::dynamic::cpu::AnyValueIsMagicFn<T>(),
          queries::ALP::dynamic::gpu::StatelessAnyValueIsMagicFn<T>(), false));

  delete data;
  return result;
}
*/



} // namespace verify_benchmarkers
