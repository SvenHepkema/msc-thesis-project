#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef RUNSPEC_HPP
#define RUNSPEC_HPP

namespace runspec {

enum UnpackerOption {
  CPU,
  STATELESS,
  STATEFUL,
  STATELESS_BRANCHLESS,
  STATEFUL_BRANCHLESS,
};

enum PatcherOption {
  NO_PATCHER,
	STATELESS_P,
	STATELESS_WITH_SCANNER_P,
	STATEFUL_P,
  NAIVE,
  NAIVE_BRANCHLESS,
  PREFETCH_POSITION,
  PREFETCH_ALL,
  PREFETCH_ALL_BRANCHLESS,
};

struct KernelSpecification {
  const UnpackerOption unpacker;
  const PatcherOption patcher;
  const unsigned n_vecs;
  const unsigned n_vals;
};

static inline const std::unordered_map<std::string, UnpackerOption>
    unpacker_options{
        {"cpu", CPU},
        {"stateless", STATELESS},
        {"stateful", STATEFUL},
        {"stateless_branchless", STATELESS_BRANCHLESS},
        {"stateful_branchless", STATEFUL_BRANCHLESS},
    };

static inline const std::unordered_map<std::string, PatcherOption>
    patcher_options{
        {"none", NO_PATCHER},
        {"stateless", STATELESS_P},
        {"stateless_with_scanner", STATELESS_WITH_SCANNER_P},
        {"stateful", STATEFUL_P},
        {"naive", NAIVE},
        {"naive_branchless", NAIVE_BRANCHLESS},
        {"prefetch_position", PREFETCH_POSITION},
        {"prefetch_all", PREFETCH_ALL},
        {"prefetch_all_branchless", PREFETCH_ALL_BRANCHLESS},
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
