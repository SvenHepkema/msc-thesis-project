#include <cstddef>
#include <cstdint>

#include "datageneration.hpp"
#include "verification.hpp"

#include "../gpu-kernels/kernels-bindings.hpp"

#include "./compressors.hpp"
#include "./decompressors.hpp"
#include "./queries.hpp"

#ifndef EXPERIMENTS_HPP
#define EXPERIMENTS_HPP

namespace experiments {

template <typename T>
verification::VerificationResult<T>
verify_fls(const runspec::RunSpecification spec) {
  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      spec.data.params, spec.data.params, spec.data.count, spec.data.count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_bp_data<T>(spec.data.name), BP_FLSCompressorFn<T>(),
          BP_FLSDecompressorFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
fls_decompress_column(const runspec::RunSpecification spec) {
  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      spec.data.params, spec.data.params, spec.data.count, spec.data.count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_bp_data<T>(spec.data.name), BP_FLSCompressorFn<T>(),
          BP_GPUDecompressorFn<T>(spec.kernel)));
}

template <typename T>
verification::VerificationResult<T>
fls_query_column(const runspec::RunSpecification spec) {
  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      spec.data.params, spec.data.params, spec.data.count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(),
          queries::cpu::FLSQueryColumnFn<T>(),
          queries::gpu::FLSQueryColumnFn<T>(spec.kernel)));
}

template <typename T>
verification::VerificationResult<T>
fls_query_column_unrolled(const runspec::RunSpecification spec) {
  return verification::run_verifier_on_parameters<T, T, int32_t,
                                                         int32_t>(
      spec.data.params, spec.data.params, spec.data.count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(), queries::cpu::FLSQueryColumnFn<T>(),
          queries::gpu::FLSQueryColumnUnrolledFn<T>(spec.kernel), false));
}

template <typename T>
verification::VerificationResult<T>
fls_compute_column(const runspec::RunSpecification spec) {
  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      spec.data.params, spec.data.params, spec.data.count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(), queries::cpu::DummyFn<T>(),
          queries::gpu::FLSComputeColumnFn<T>(spec.kernel)));
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
alp_decompress_column(const runspec::RunSpecification spec) {
  auto [data, generator] =
      data::lambda::get_alp_reusable_datastructure<T>(spec.data);

  auto result =
      verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                               int32_t, int32_t>(
          spec.data.params, spec.data.params, spec.data.count, spec.data.count,
          verification::get_equal_decompression_verifier<
              T, alp::AlpCompressionData<T>, int32_t, int32_t>(
              generator, ALP_FLSDecompressorFn<T>(),
              ALP_GPUDecompressorFn<T>(spec.kernel), false));

  delete data;
  return result;
}

template <typename T>
verification::VerificationResult<T>
alp_query_column(const runspec::RunSpecification spec) {
  auto [data, generator] =
      data::lambda::get_reusable_compressed_binary_column<T>(spec.data);

  auto result =
      verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                               int32_t, int32_t>(
          spec.data.params, spec.data.params, spec.data.count, 1,
          verification::get_equal_decompression_verifier<
              T, alp::ALPMagicCompressionData<T>, int32_t, int32_t>(
              generator, queries::cpu::ALPQueryColumnFn<T>(),
              queries::gpu::ALPQueryColumnFn<T>(spec.kernel), false));

  delete data;
  return result;
}

} // namespace experiments

#endif // EXPERIMENTS_HPP
