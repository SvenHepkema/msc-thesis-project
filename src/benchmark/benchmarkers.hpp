#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../verification/datageneration.hpp"
#include "../verification/verification.hpp"

#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "../gpu-alp/alp-benchmark-kernels-bindings.hpp"
#include "../gpu-alp/alp-test-kernels-bindings.hpp"
#include "../gpu-fls/fls-benchmark-kernels-bindings.hpp"

#include "../verification/queries.h"

namespace benchmarkers {

template <typename T>
verification::VerificationResult<T>
bench_int_baseline(const size_t a_count,
                   [[maybe_unused]] const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(1, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(),
          queries::baseline::cpu::IntAnyValueIsMagicFn<T>(),
          queries::baseline::gpu::IntAnyValueIsMagicFn<T>()));
}

template <typename T>
verification::VerificationResult<T> bench_old_fls_contains_zero_value_bitwidths(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(1, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(),
          queries::FLS::cpu::AnyValueIsMagicFn<T>(),
          queries::FLS::gpu::OldAnyValueIsMagicFn<T>()));
}

template <typename T>
verification::VerificationResult<T> bench_bp_contains_zero_value_bitwidths(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(1, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(),
          queries::FLS::cpu::AnyValueIsMagicFn<T>(),
          queries::FLS::gpu::StatelessAnyValueIsMagicFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
bench_bp_stateful_contains_zero_value_bitwidths(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(1, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(),
          queries::FLS::cpu::AnyValueIsMagicFn<T>(),
          queries::FLS::gpu::StatefulAnyValueIsMagicFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
bench_float_baseline(const size_t a_count,
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
bench_alp_ec_stateless(const size_t a_count,
                       [[maybe_unused]] const std::string dataset_name) {
  auto exception_percentages =
      verification::generate_integer_range<int32_t>(0, 70);
  auto [data, generator] = data::lambda::get_alp_reusable_datastructure<T>(
      "exceptions_per_vec", a_count);

  auto result =
      verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                               int32_t, int32_t>(
          exception_percentages, exception_percentages, a_count, 1,
          verification::get_equal_decompression_verifier<
              T, alp::AlpCompressionData<T>, int32_t, int32_t>(
              generator, queries::ALP::constant::cpu::AnyValueIsMagicFn<T>(),
              queries::ALP::constant::gpu::StatelessAnyValueIsMagicFn<T>(),
              false));

  delete data;

  return result;
}

template <typename T>
verification::VerificationResult<T>
bench_alp_ec_stateful(const size_t a_count,
                      [[maybe_unused]] const std::string dataset_name) {
  auto exception_percentages =
      verification::generate_integer_range<int32_t>(0, 70);
  auto [data, generator] = data::lambda::get_alp_reusable_datastructure<T>(
      "exceptions_per_vec", a_count);

  auto result =
      verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                               int32_t, int32_t>(
          exception_percentages, exception_percentages, a_count, 1,
          verification::get_equal_decompression_verifier<
              T, alp::AlpCompressionData<T>, int32_t, int32_t>(
              generator, queries::ALP::constant::cpu::AnyValueIsMagicFn<T>(),
              queries::ALP::constant::gpu::StatefulAnyValueIsMagicFn<T>(),
              false));

  delete data;

  return result;
}

template <typename T>
verification::VerificationResult<T> bench_alp_ec_stateful_extended(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  auto exception_percentages =
      verification::generate_integer_range<int32_t>(0, 70);
  auto [data, generator] = data::lambda::get_alp_reusable_datastructure<T>(
      "exceptions_per_vec", a_count);

  auto result = verification::run_verifier_on_parameters<
      T, alp::AlpCompressionData<T>, int32_t, int32_t>(
      exception_percentages, exception_percentages, a_count, 1,
      verification::get_equal_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          generator, queries::ALP::constant::cpu::AnyValueIsMagicFn<T>(),
          queries::ALP::constant::gpu::StatefulExtendedAnyValueIsMagicFn<T>(),
          false));

  delete data;

  return result;
}

template <typename T>
verification::VerificationResult<T>
bench_alp_vbw_stateless(const size_t a_count,
                        [[maybe_unused]] const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 4);

  auto [data, generator] = data::lambda::get_alp_reusable_datastructure<T>(
      "value_bit_width", a_count);

  auto result =
      verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                               int32_t, int32_t>(
          value_bit_widths, value_bit_widths, a_count, 1,
          verification::get_equal_decompression_verifier<
              T, alp::AlpCompressionData<T>, int32_t, int32_t>(
              generator, queries::ALP::constant::cpu::AnyValueIsMagicFn<T>(),
              queries::ALP::constant::gpu::StatelessAnyValueIsMagicFn<T>(),
              false));

  delete data;
  return result;
}

template <typename T>
verification::VerificationResult<T>
bench_alp_vbw_stateful(const size_t a_count,
                       [[maybe_unused]] const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 4);

  auto [data, generator] = data::lambda::get_alp_reusable_datastructure<T>(
      "value_bit_width", a_count);

  auto result =
      verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                               int32_t, int32_t>(
          value_bit_widths, value_bit_widths, a_count, 1,
          verification::get_equal_decompression_verifier<
              T, alp::AlpCompressionData<T>, int32_t, int32_t>(
              generator, queries::ALP::constant::cpu::AnyValueIsMagicFn<T>(),
              queries::ALP::constant::gpu::StatefulAnyValueIsMagicFn<T>(),
              false));

  delete data;
  return result;
}

template <typename T>
verification::VerificationResult<T> bench_alp_vbw_stateful_extended(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 4);

  auto [data, generator] = data::lambda::get_alp_reusable_datastructure<T>(
      "value_bit_width", a_count);

  auto result = verification::run_verifier_on_parameters<
      T, alp::AlpCompressionData<T>, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          generator, queries::ALP::constant::cpu::AnyValueIsMagicFn<T>(),
          queries::ALP::constant::gpu::StatefulExtendedAnyValueIsMagicFn<T>(),
          false));

  delete data;
  return result;
}

} // namespace benchmarkers
