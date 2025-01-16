#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "functions.h"
#include "verification/verification.hpp"
#include "common/runspec.hpp"


struct CLIArgs {
  std::string verifier;
  int32_t datatype_width;
  const runspec::RunSpecification runspec;
  bool print_debug;

  CLIArgs(std::string a_verifier, int32_t a_datatype_width,
          const runspec::RunSpecification a_runspecification, bool a_print_debug)
      : verifier(a_verifier), datatype_width(a_datatype_width),
        runspec(a_runspecification), print_debug(a_print_debug) {}
};

static CLIArgs parse_cli_args(int argc, char **argv) {
  if (argc != 6 && argc != 7) {
    throw std::invalid_argument("Not the required amount of arguments.");
  }
  int32_t argcounter = 0;

  std::string verifier = argv[++argcounter];
  int32_t datatype_width = std::atoi(argv[++argcounter]);
  std::string dataset_name = argv[++argcounter];

  std::string kernelspec = (argc == 7) ? argv[++argcounter] : "cpu";
  size_t n_vecs = static_cast<size_t>(std::atoi(argv[++argcounter]));
  size_t count = n_vecs * 1024;
  bool print_debug = std::atoi(argv[++argcounter]);

  return CLIArgs(verifier, datatype_width,
                 runspec::RunSpecification(count, dataset_name, kernelspec),
                 print_debug);
}

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
          difference.template log<T>();
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
      functions::Fastlanes<T>::functions.at(args.verifier)(args.runspec.count,
                                                           args.runspec.dataset_name);
  return process_results<T>(results, args);
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
int32_t run_verifier(CLIArgs args) {
  verification::VerificationResult<T> results = functions::Alp<T>::functions.at(
      args.verifier)(args.runspec.count, args.runspec.dataset_name);
  return process_results<T>(results, args);
}

int main(int argc, char **argv) {
  CLIArgs args =parse_cli_args(argc, argv);

  if (functions::Fastlanes<uint32_t>::functions.find(args.verifier) !=
      functions::Fastlanes<uint32_t>::functions.end()) {
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
  } else if (functions::Alp<float>::functions.find(args.verifier) !=
             functions::Alp<float>::functions.end()) {
    switch (args.datatype_width) {
    case 64: {
      return run_verifier<double>(args);
    }
    case 32: {
      return run_verifier<float>(args);
    }
    }
  } else {
    throw std::invalid_argument("This function is not supported");
  }
}
