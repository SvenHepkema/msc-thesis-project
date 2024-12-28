#include <functional>

#include "benchmark/benchmarkers.hpp"
#include "verification/verification.hpp"
#include "verification/verifiers.hpp"

namespace functions {

template <typename T>
using Verifier = std::function<verification::VerificationResult<T>(
    const size_t, const std::string)>;

template <class T> struct Fastlanes {
  static inline const std::unordered_map<std::string, Verifier<T>> functions = {
      {"bp",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_bitpacking<T>(count, dataset_name);
       }},
      {"gpu_bp_stateless",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_bp_stateless<T>(count, dataset_name);
       }},
      {"gpu_bp_stateless_multivec",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_bp_stateful_multivec<T>(count, dataset_name);
       }},
      {"gpu_bp_stateful",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_bp_stateful<T>(count, dataset_name);
       }},
      {"ffor",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_ffor<T>(count, dataset_name);
       }},
      {"gpu_unffor",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_unffor<T>(count, dataset_name);
       }},
      {"bench_int_baseline",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_int_baseline<T>(
             count, dataset_name);
       }},
      {"bench_old_fls_vbw",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_old_fls_contains_zero_value_bitwidths<T>(
             count, dataset_name);
       }},
      {"bench_bp_vbw",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_bp_contains_zero_value_bitwidths<T>(
             count, dataset_name);
       }},
      {"bench_bp_stateful_vbw",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_bp_stateful_contains_zero_value_bitwidths<T>(
             count, dataset_name);
       }},
  };
};

template <class T> struct Alp {
  static inline std::unordered_map<std::string, Verifier<T>> functions = {
      {"alp",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_alp<T>(count, dataset_name);
       }},
     {"gpu_alp_stateless",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_alp_stateless<T>(count, dataset_name);
       }},
      {"gpu_alp_stateful",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_alp_stateful<T>(count, dataset_name);
       }},
      {"gpu_alp_extended_state",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_alp_stateful_extended<T>(count, dataset_name);
       }},
      {"gpu_alp_extended_state_2vec",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_alp_stateful_extended_multivec<T, 2>(count, dataset_name);
       }},
      {"gpu_alp_extended_state_4vec",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_alp_stateful_extended_multivec<T, 4>(count, dataset_name);
       }},
      {"alprd",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_alprd<T>(count, dataset_name);
       }},
      {"gpu_alprd",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_alprd<T>(count, dataset_name);
       }},
      {"bench_float_baseline",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_float_baseline<T>(
             count, dataset_name);
       }},
      {"bench_alp_exception_count",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_alp_varying_exception_count<T>(count, dataset_name);
       }},
      {"bench_alp_value_bit_width",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_alp_varying_value_bit_width<T>(count, dataset_name);
       }},
      {"bench_alp_multiple_columns",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_alp_multiple_columns<T>(count, dataset_name);
       }},
  };
};

} // namespace functions
