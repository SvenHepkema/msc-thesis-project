#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "verification/verification.hpp"
#include "verification/verifiers.hpp"

struct CLIArgs {
	std::string compression_type;
  size_t count;
  int32_t lane_width;
  std::string verifier;
  bool use_random_data;
  bool print_debug;

  CLIArgs(int argc, char **argv) {
    if (argc != 7) {
      throw std::invalid_argument("Not the required amount of arguments.");
    }
    int32_t argcounter = 0;

    compression_type = argv[++argcounter];
    size_t n_vecs = static_cast<size_t>(std::atoi(argv[++argcounter]));
    count = n_vecs * 1024;
    lane_width = std::atoi(argv[++argcounter]);
    verifier = argv[++argcounter];
    use_random_data = std::atoi(argv[++argcounter]);
    print_debug = std::atoi(argv[++argcounter]);
  }
};

template <typename T>
int32_t process_results(verification::VerificationResult<T> results,
                        CLIArgs args) {
  if (results.size() == 0) {
    if (args.print_debug) {
      fprintf(stderr, "Compression successful.\n");
    }
    return 0;
  }

  if (args.print_debug) {
    for (auto result : results) {
      fprintf(stderr, "\nValue bit width %d failed.\n", result.first);

      for (auto difference : result.second) {
        difference.template log<double>();
      }
    }
    fprintf(stderr, "\n[%ld/%d] Value bit widths failed.\n", results.size(),
            int32_t{sizeof(T)} * 8);
  }

  return static_cast<int32_t>(results.size());
}

template <typename T> int32_t run_fls_verification(CLIArgs args) {
  verification::VerificationResult<T> results = verifiers::get_fls_verifier<T>(
      args.verifier)(args.count, args.use_random_data);
  return process_results<T>(results, args);
}

template <typename T> int32_t run_alp_verification(CLIArgs args) {
  verification::VerificationResult<T> results = verifiers::get_alp_verifier<T>(
      args.verifier)(args.count, args.use_random_data);
  return process_results<T>(results, args);
}

int main(int argc, char **argv) {
  CLIArgs args(argc, argv);

  if (args.compression_type == "fls") {
#ifdef DATA_TYPE
    // return run_fls_verification<DATA_TYPE>(args);
#else
    switch (args.lane_width) {
    case 64: {
      return run_fls_verification<uint64_t>(args);
    }
    case 32: {
      return run_fls_verification<uint32_t>(args);
    }
    case 16: {
      return run_fls_verification<uint16_t>(args);
    }
    case 8: {
      return run_fls_verification<uint8_t>(args);
    }
    }
#endif
  } else if (args.compression_type == "alp") {
#ifdef DATA_TYPE
    return run_alp_verification<DATA_TYPE>(args);
#else
    switch (args.lane_width) {
    case 64: {
      return run_alp_verification<double>(args);
    }
    case 32: {
      return run_alp_verification<float>(args);
    }
    }
#endif
  } else {
    throw std::invalid_argument("This compression type is not supported");
	}
}
