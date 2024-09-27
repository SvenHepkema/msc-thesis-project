#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "verification/verification.hpp"
#include "verification/verifiers.hpp"

struct CLIArgs {
  std::string verifier;
  int32_t datatype_width;
  std::string dataset_name;
  size_t count;
  bool print_debug;

  CLIArgs(int argc, char **argv) {
    if (argc != 6) {
      throw std::invalid_argument("Not the required amount of arguments.");
    }
    int32_t argcounter = 0;

    verifier = argv[++argcounter];
    datatype_width = std::atoi(argv[++argcounter]);
    dataset_name = argv[++argcounter];
    size_t n_vecs = static_cast<size_t>(std::atoi(argv[++argcounter]));
    count = n_vecs * 1024;
    print_debug = std::atoi(argv[++argcounter]);
  }
};

template <typename T>
int32_t process_results(verification::VerificationResult<T> results,
                        CLIArgs args) {

  int32_t runs_failed = 0;
  for (size_t i{0}; i < results.size(); i++) {
    if (!results[i].success) {
      ++runs_failed;

      if (args.print_debug) {
        fprintf(stderr, "\n Run %lu failed.\n", i);

        for (auto difference : results[i].differences) {
          difference.template log<double>();
        }
      }
    }
  }

  if (args.print_debug) {
    if (runs_failed == 0) {
      fprintf(stderr, "Compression successful.\n");
    } else {
      fprintf(stderr, "\n[%d/%ld] Runs failed.\n", runs_failed, results.size());
    }
  }

  return runs_failed;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
int32_t run_verifier(CLIArgs args) {
  verification::VerificationResult<T> results =
      verifiers::Fastlanes<T>::verifiers.at(args.verifier)(args.count,
                                                           args.dataset_name);
  return process_results<T>(results, args);
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
int32_t run_verifier(CLIArgs args) {
  verification::VerificationResult<T> results = verifiers::Alp<T>::verifiers.at(
      args.verifier)(args.count, args.dataset_name);
  return process_results<T>(results, args);
}

int main(int argc, char **argv) {
  CLIArgs args(argc, argv);

  if (verifiers::Fastlanes<uint32_t>::verifiers.find(args.verifier) !=
      verifiers::Fastlanes<uint32_t>::verifiers.end()) {
    switch (args.datatype_width) {
    case 64: {
      return run_verifier<uint64_t>(args);
    }
    case 32: {
      return run_verifier<uint32_t>(args);
    }
    case 16: {
      return run_verifier<uint16_t>(args);
    }
    case 8: {
      return run_verifier<uint8_t>(args);
    }
    }
  } else if (verifiers::Alp<float>::verifiers.find(args.verifier) !=
             verifiers::Alp<float>::verifiers.end()) {
    switch (args.datatype_width) {
    case 64: {
      return run_verifier<double>(args);
    }
    case 32: {
      return run_verifier<float>(args);
    }
    }
  } else {
    throw std::invalid_argument("This verifier is not supported");
  }
}
