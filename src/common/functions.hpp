#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "../engine/benchmarkers.hpp"
#include "../engine/verification.hpp"
#include "../engine/verifiers.hpp"

#include "../gpu-kernels/kernels-bindings.hpp"

#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

namespace functions {

template <typename T>
using Verifier = std::function<verification::VerificationResult<T>(
    const size_t, const std::string)>;

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
      OPTION("verify_fls_bp", verifiers::verify_fls_bp),
      OPTION("verify_gpu_bp", verifiers::verify_gpu_bp),
      OPTION("bench_bp_vbw", benchmarks::bench_bp_vbw),
  };
};

template <class T> struct Alp {
  static inline std::unordered_map<std::string, Verifier<T>> functions = {
      OPTION("verify_alp", verifiers::verify_alp),
      OPTION("verify_gpu_alp", verifiers::verify_gpu_alp),
      OPTION("verify_magic_query_alp", verifiers::verify_magic_query_alp),
      OPTION("bench_alp_vbw", benchmarks::bench_alp_vbw),
  };
};
} // namespace functions
#endif // FUNCTIONS_HPP
