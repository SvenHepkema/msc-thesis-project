#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "verification.h"
#include "verifiers.h"

struct CLIArgs {
  size_t count;
  int32_t lane_width;
  std::string compression_type;
  bool use_random_data;
  bool print_debug;

  CLIArgs(int argc, char **argv) {
    if (argc != 6) {
      throw std::invalid_argument("Not the required amount of arguments.");
    }
    int32_t argcounter = 0;

    size_t n_vecs = static_cast<size_t>(std::atoi(argv[++argcounter]));
    count = n_vecs * 1024;
    lane_width = std::atoi(argv[++argcounter]);
    compression_type = argv[++argcounter];
    use_random_data = std::atoi(argv[++argcounter]);
    print_debug = std::atoi(argv[++argcounter]);
  }
};

template <typename T> int run_verification(CLIArgs args) {
  verification::VerificationResult<T> results = verifiers::get_verifier<T>(
      args.compression_type)(args.count, args.use_random_data);

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
        difference.log();
      }
    }
    fprintf(stderr, "\n[%ld/%d] Value bit widths failed.\n", results.size(),
            int32_t{sizeof(T)} * 8);
  }

  return 1;
}

int main(int argc, char **argv) {
  CLIArgs args(argc, argv);

#ifdef DATA_TYPE
  return run_verification<DATA_TYPE>(args);
#else
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
#endif
}
