#include <functional>

#include "benchmark/benchmarkers.hpp"
#include "benchmark/verify-benchmarkers.hpp"
#include "verification/verification.hpp"
#include "verification/verifiers.hpp"

namespace functions {

template <typename T>
using Verifier = std::function<verification::VerificationResult<T>(
    const size_t, const std::string)>;

template <class T> struct Fastlanes {
  static inline const std::unordered_map<std::string, Verifier<T>> functions = {
      {"verify_fls_bp",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_bitpacking<T>(count, dataset_name);
       }},
      {"verify_fls_ffor",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_ffor<T>(count, dataset_name);
       }},
      {"verify_bp",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_bp<T>(count, dataset_name);
       }},
      {"bench_bp_vbw",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_bp_contains_zero_value_bitwidths<T>(
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
         return verifiers::verify_gpu_alp_stateful_extended<T>(count,
                                                               dataset_name);
       }},
      {"gpu_alp_extended_state_2vec",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_alp_stateful_extended_multivec<T, 2>(
             count, dataset_name);
       }},
      {"gpu_alp_extended_state_4vec",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_gpu_alp_stateful_extended_multivec<T, 4>(
             count, dataset_name);
       }},
      {"bench_float_baseline",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_float_baseline<T>(count, dataset_name);
       }},
      {"bench_alp_ec_stateless",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_alp_ec_stateless<T>(count, dataset_name);
       }},
      {"bench_alp_ec_stateful",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_alp_ec_stateful<T>(count, dataset_name);
       }},
      {"bench_alp_ec_stateful_extended",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_alp_ec_stateful_extended<T>(count,
                                                                dataset_name);
       }},
      {"bench_alp_vbw_stateless",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_alp_vbw_stateless<T>(count, dataset_name);
       }},
      {"bench_alp_vbw_stateful",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_alp_vbw_stateful<T>(count, dataset_name);
       }},
      {"bench_alp_vbw_stateful_extended",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return benchmarkers::bench_alp_vbw_stateful_extended<T>(count,
                                                                 dataset_name);
       }},
      {"verify_bench_alp_stateless",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verify_benchmarkers::verify_bench_alp_stateless<T>(count, dataset_name);
       }},
      {"verify_bench_alp_stateful",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verify_benchmarkers::verify_bench_alp_stateful<T>(count, dataset_name);
       }},
      {"verify_bench_alp_stateful_extended",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verify_benchmarkers::verify_bench_alp_stateful_extended<T>(count,
                                                                 dataset_name);
       }},
  };
};

} // namespace functions
