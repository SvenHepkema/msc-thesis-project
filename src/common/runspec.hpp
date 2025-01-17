#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef RUNSPEC_HPP
#define RUNSPEC_HPP

namespace runspec {

// The divider is used to do some logic
enum KernelOption {
  TEST_STATELESS_1_1,
  TEST_STATEFUL_1_1,
  TEST_STATELESS_BRANCHLESS_1_1,
  TEST_STATEFUL_BRANCHLESS_1_1,
  QUERY_STATELESS_1_1,
  // ===================
  DIVIDER_FLS_ALP_KERNELS,
  // ===================
};

enum VerifierOption {
	FLSDECOMPRESSOR,
  // ===================
  DIVIDER_FLS_DECOMPRESSION_VERIFIERS,
  // ===================
  // ===================
  DIVIDER_FLS_ALP_VERIFIERS,
  // ===================
  // ===================
  DIVIDER_ALP_DECOMPRESSION_VERIFIERS,
  // ===================
};

struct KernelSpecification {
  const KernelOption kernel;

  const unsigned n_vectors;
  const unsigned n_values;

  const VerifierOption verifier;

  bool verifier_is_decompressor() const {
    return ((DIVIDER_FLS_DECOMPRESSION_VERIFIERS < verifier) &&
            (verifier < DIVIDER_FLS_ALP_VERIFIERS)) ||
           (DIVIDER_ALP_DECOMPRESSION_VERIFIERS < verifier);
  }
};

static inline const std::unordered_map<std::string, KernelSpecification>
    kernel_options{
        {"test_stateless_1_1", KernelSpecification{TEST_STATELESS_1_1, 1, 1,
                                                   DIVIDER_FLS_ALP_VERIFIERS}},
        {"test_stateful_1_1", KernelSpecification{TEST_STATEFUL_1_1, 1, 1,
                                                  DIVIDER_FLS_ALP_VERIFIERS}},
        {"test_stateless_branchless_1_1",
         KernelSpecification{TEST_STATELESS_BRANCHLESS_1_1, 1, 1,
                             DIVIDER_FLS_ALP_VERIFIERS}},
        {"test_stateful_branchless_1_1",
         KernelSpecification{TEST_STATEFUL_BRANCHLESS_1_1, 1, 1,
                             DIVIDER_FLS_ALP_VERIFIERS}},
        {"query_stateless_1_1", KernelSpecification{QUERY_STATELESS_1_1, 1, 1,
                                                    DIVIDER_FLS_ALP_VERIFIERS}},
    };

enum DataGenerationParametersType {
  NONE,
  EC,
  VBW,
};

struct DataSpecification {
  const size_t count;
  const DataGenerationParametersType params_type;
  const std::vector<int32_t> params;
  const std::string name;

  bool use_index() const { return name == "index"; }

  bool use_file() const { return name != "index" && name != "random"; }
};

struct RunSpecification {
  const std::string function;
  const KernelSpecification kernel;
  const DataSpecification data;
};

} // namespace runspec

#endif // RUNSPEC_HPP
