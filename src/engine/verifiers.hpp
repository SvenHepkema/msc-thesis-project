#include <cstddef>
#include <cstdint>

#include "datageneration.hpp"
#include "verification.hpp"

#include "../gpu-kernels/kernels-bindings.hpp"

#include "./compressors.h"
#include "./decompressors.h"
#include "./queries.h"

#ifndef VERIFIERS_HPP
#define VERIFIERS_HPP

namespace verifiers {

template <typename T>
verification::VerificationResult<T>
verify_fls_bp(const runspec::RunSpecification spec) {
  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      spec.data.params, spec.data.params, spec.data.count, spec.data.count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_bp_data<T>(spec.data.name), BP_FLSCompressorFn<T>(),
          BP_FLSDecompressorFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_bp(const runspec::RunSpecification spec) {
  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      spec.data.params, spec.data.params, spec.data.count, spec.data.count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_bp_data<T>(spec.data.name), BP_FLSCompressorFn<T>(),
          BP_GPUDecompressorFn<T>(spec.kernel)));
}

template <typename T>
verification::VerificationResult<T>
verify_alp(const runspec::RunSpecification spec) {
  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      spec.data.params, spec.data.params, spec.data.count, spec.data.count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(spec.data.name),
          ALP_FLSCompressorFn<T>(), ALP_FLSDecompressorFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alp(const runspec::RunSpecification spec) {
  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      spec.data.params, spec.data.params, spec.data.count, spec.data.count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(spec.data.name),
          ALP_FLSCompressorFn<T>(), ALP_GPUDecompressorFn<T>(spec.kernel)));
}

template <typename T>
verification::VerificationResult<T>
verify_magic_query_alp(const runspec::RunSpecification spec) {
  auto [data, generator] =
      data::lambda::get_reusable_compressed_binary_column<T>(
          spec.data.params_type, spec.data.count);

  auto result = verification::run_verifier_on_parameters<
      T, alp::ALPMagicCompressionData<T>, int32_t, int32_t>(
      spec.data.params, spec.data.params, spec.data.count, 1,
      verification::get_equal_decompression_verifier<
          T, alp::ALPMagicCompressionData<T>, int32_t, int32_t>(
          generator, queries::cpu::ALPAnyValueIsMagicFn<T>(),
          queries::gpu::ALPAnyValueIsMagicFn<T>(spec.kernel), false));

  delete data;
  return result;
}

} // namespace verifiers

#endif // VERIFIERS_HPP
