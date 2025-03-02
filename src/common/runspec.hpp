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
  DUMMY,
	OLD_FLS_ADJUSTED,
  NON_INTERLEAVED,
  STATELESS,
  STATEFUL_CACHE,
  STATEFUL_LOCAL_MEMORY_1,
  STATEFUL_LOCAL_MEMORY_2,
  STATEFUL_LOCAL_MEMORY_4,
  STATEFUL_REGISTER_1,
  STATEFUL_REGISTER_2,
  STATEFUL_REGISTER_4,
  STATEFUL_REGISTER_BRANCHLESS_1,
  STATEFUL_REGISTER_BRANCHLESS_2,
  STATEFUL_REGISTER_BRANCHLESS_4,
  STATELESS_BRANCHLESS,
  STATEFUL_BRANCHLESS,
};

enum PatcherOption {
  NO_PATCHER,
  STATELESS_P,
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
        {"dummy", DUMMY},
        {"old_fls_adjusted", OLD_FLS_ADJUSTED},
        {"noninterleaved", NON_INTERLEAVED},
        {"stateless", STATELESS},
        {"stateful-cache", STATEFUL_CACHE},
        {"stateful-local-1", STATEFUL_LOCAL_MEMORY_1},
        {"stateful-local-2", STATEFUL_LOCAL_MEMORY_2},
        {"stateful-local-4", STATEFUL_LOCAL_MEMORY_4},
        {"stateful-register-1", STATEFUL_REGISTER_1},
        {"stateful-register-2", STATEFUL_REGISTER_2},
        {"stateful-register-4", STATEFUL_REGISTER_4},
        {"stateful-register-branchless-1", STATEFUL_REGISTER_BRANCHLESS_1},
        {"stateful-register-branchless-2", STATEFUL_REGISTER_BRANCHLESS_2},
        {"stateful-register-branchless-4", STATEFUL_REGISTER_BRANCHLESS_4},
        {"stateless_branchless", STATELESS_BRANCHLESS},
        {"stateful_branchless", STATEFUL_BRANCHLESS},
    };

static inline const std::unordered_map<std::string, PatcherOption>
    patcher_options{
        {"none", NO_PATCHER},
        {"stateless", STATELESS_P},
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
  const size_t n_vecs;
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
