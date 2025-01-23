#include <cstddef>
#include <cstdint>

#include "../engine/datageneration.hpp"
#include "../engine/verification.hpp"

#include "../alp/alp-bindings.hpp"

#include "../engine/queries.h"

#ifndef BENCHMARKS_HPP
#define BENCHMARKS_HPP

namespace benchmarks {

template <typename T>
verification::VerificationResult<T>
bench_bp_vbw(const runspec::RunSpecification spec) {
  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      spec.data.params, spec.data.params, spec.data.count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(),
          queries::cpu::FLSAnyValueIsMagicFn<T>(),
          queries::gpu::FLSAnyValueIsMagicFn<T>(spec.kernel)));
}

template <typename T>
verification::VerificationResult<T>
bench_alp_vbw(const runspec::RunSpecification spec) {
  auto [data, generator] =
      data::lambda::get_reusable_compressed_binary_column<T>(
          spec.data.params_type, spec.data.count);

  auto result =
      verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                               int32_t, int32_t>(
          spec.data.params, spec.data.params, spec.data.count, 1,
          verification::get_equal_decompression_verifier<
              T, alp::ALPMagicCompressionData<T>, int32_t, int32_t>(
              generator, queries::cpu::ALPAnyValueIsMagicFn<T>(),
              queries::gpu::ALPAnyValueIsMagicFn<T>(spec.kernel), false));

  delete data;
  return result;
}

} // namespace benchmarks

#endif // BENCHMARKS_HPP
