#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "verification.h"

struct CLIArgs {
  size_t count;
  int32_t lane_width;
  std::string compression_type;
  bool use_random_data;
  bool use_signed;
  bool print_debug;

  CLIArgs(int argc, char **argv) {
    if (argc != 7) {
      throw std::invalid_argument("Not the required amount of arguments.");
    }
    int32_t argcounter = 0;

    size_t n_vecs = std::atoi(argv[++argcounter]);
    count = n_vecs * 1024;
    lane_width = std::atoi(argv[++argcounter]);
    compression_type = argv[++argcounter];
    use_random_data = std::atoi(argv[++argcounter]);
    use_signed = std::atoi(argv[++argcounter]);
    print_debug = std::atoi(argv[++argcounter]);
  }
};

template <typename T>
std::function<verification::VerificationResult<T>(const size_t, const bool)>
get_verifier(CLIArgs args) {
  if (args.compression_type == "bp") {
    return
        [](const int32_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verification::verify_bitpacking<T>(count, use_random_data);
        };
  } else if (args.compression_type == "ffor") {
    return
        [](const int32_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verification::verify_ffor<T>(count, use_random_data);
        };
  } else {
    throw std::invalid_argument("This compression type is not supported");
  }
}

template <typename T> int run_verification(CLIArgs args) {
  verification::VerificationResult<T> results =
      get_verifier<T>(args)(args.count, args.use_random_data);

  if (results.size() == 0) {
    if (args.print_debug)
      fprintf(stderr, "Compression successful.\n");
    return 0;
  }

  if (args.print_debug)
    for (auto result : results) {
      fprintf(stderr, "\nValue bit width %d failed.\n", result.first);

      for (auto difference : result.second) {
        difference.log();
      }
    };
  fprintf(stderr, "\n[%d/%d] Value bit widths failed.\n",
          (int32_t)results.size(), (int32_t)sizeof(T) * 8);

  return 1;
}

int main(int argc, char **argv) {
  CLIArgs args(argc, argv);

#ifdef FAST_COMPILATION
	return run_verification<uint64_t>(args);
#else
  if (args.use_signed) {
    switch (args.lane_width) {
    case 64: {
      return run_verification<int64_t>(args);
    }
    case 32: {
      return run_verification<int32_t>(args);
    }
    case 16: {
      return run_verification<int16_t>(args);
    }
    case 8: {
      return run_verification<int8_t>(args);
    }
    }
  } else {
    switch (args.lane_width) {
    case 64: {
      return run_verification<uint64_t>(args);
    }
    case 32: {
      return run_verification<uint32_t>(args);
    }
    case 16: {
      return run_verification<uint16_t>(args);
    }
    case 8: {
      return run_verification<uint8_t>(args);
    }
    }
	}
#endif

    return 0;
  }
