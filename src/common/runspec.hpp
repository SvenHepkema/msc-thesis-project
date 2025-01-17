#include <cstddef>
#include <cstdint>
#include <string>
#include <functional>

#include "../engine/benchmarkers.hpp"
#include "../engine/verify-benchmarkers.hpp"
#include "../engine/verification.hpp"
#include "../engine/verifiers.hpp"

#include "../gpu-kernels/fls-kernels-bindings.hpp"

#ifndef RUNSPEC_HPP
#define RUNSPEC_HPP

namespace runspec {

struct RunSpecification {
  const size_t count;
  const std::string dataset_name;
  const kernels::KernelSpecification spec;

  RunSpecification() : count(1024), dataset_name("random"), spec(kernels::KernelSpecification()) {}

  RunSpecification(const size_t a_count, const std::string a_dataset_name,
                   const std::string kernel)
      : count(a_count), dataset_name(a_dataset_name),
        spec(kernels::kernel_options.at(kernel)) {}
};

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
}

#endif // RUNSPEC_HPP
