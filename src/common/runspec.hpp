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
	CPU,
  STATELESS_1_1,
  STATELESS_4_1,
  STATEFUL_1_1,
  STATEFUL_4_1,
  STATELESS_BRANCHLESS_1_1,
  STATELESS_BRANCHLESS_4_1,
  STATEFUL_BRANCHLESS_1_1,
  STATEFUL_BRANCHLESS_4_1,
};

struct KernelSpecification {
  const KernelOption kernel;

  const unsigned n_vectors;
  const unsigned n_values;
};

static inline const std::unordered_map<std::string, KernelSpecification>
    kernel_options{
        {"cpu", KernelSpecification{CPU, 1, 1}},
        {"stateless_1_1", KernelSpecification{STATELESS_1_1, 1, 1}},
        {"stateless_4_1", KernelSpecification{STATELESS_4_1, 4, 1}},
        {"stateful_1_1", KernelSpecification{STATEFUL_1_1, 1, 1}},
        {"stateful_4_1", KernelSpecification{STATEFUL_4_1, 4, 1}},
        {"stateless_branchless_1_1",
         KernelSpecification{STATELESS_BRANCHLESS_1_1, 1, 1}},
        {"stateless_branchless_4_1",
         KernelSpecification{STATELESS_BRANCHLESS_4_1, 4, 1}},
        {"stateful_branchless_1_1",
         KernelSpecification{STATEFUL_BRANCHLESS_1_1, 1, 1}},
        {"stateful_branchless_4_1",
         KernelSpecification{STATEFUL_BRANCHLESS_4_1, 4, 1}},
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
