#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "../engine/verification.hpp"
#include "../engine/experiments.hpp"

#include "../gpu-kernels/kernels-bindings.hpp"

#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

namespace functions {

template <typename T>
using Verifier = std::function<verification::VerificationResult<T>(
    const runspec::RunSpecification)>;

#define OPTION(NAME, FUNCTION)                                                 \
  {                                                                            \
    NAME,                                                                      \
        [](const runspec::RunSpecification spec)                               \
            -> verification::VerificationResult<T> {                           \
          return FUNCTION<T>(spec);                                            \
        }                                                                      \
  }

template <class T> struct Fastlanes {
  static inline const std::unordered_map<std::string, Verifier<T>> functions = {
      OPTION("verify_fls", experiments::verify_fls),
      OPTION("fls_decompress", experiments::fls_decompress_column),
      OPTION("fls_query", experiments::fls_query_column),
      OPTION("fls_compute", experiments::fls_compute_column),
  };
};

template <class T> struct Alp {
  static inline std::unordered_map<std::string, Verifier<T>> functions = {
      OPTION("verify_alp", experiments::verify_alp),
      OPTION("alp_decompress", experiments::alp_decompress_column),
      OPTION("alp_query", experiments::alp_query_column),
  };
};
} // namespace functions
#endif // FUNCTIONS_HPP
