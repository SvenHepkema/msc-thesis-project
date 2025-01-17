#include <cstddef>
#include <cstdint>
#include <string>
#include <functional>

#include "../engine/benchmarkers.hpp"
#include "../engine/verify-benchmarkers.hpp"
#include "../engine/verification.hpp"
#include "../engine/verifiers.hpp"

#include "../gpu-kernels/kernels-bindings.hpp"

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
  };
};

template <class T> struct Alp {
  static inline std::unordered_map<std::string, Verifier<T>> functions = {
      {"alp",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verifiers::verify_alp<T>(count, dataset_name);
       }},
  };
};
}

#endif // RUNSPEC_HPP
