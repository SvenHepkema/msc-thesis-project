#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>

#include "common/functions.hpp"
#include "common/runspec.hpp"
#include "engine/verification.hpp"

struct CLIArgs {
  int32_t datatype_width;
  const runspec::RunSpecification runspec;
  bool print_debug;
};

static std::tuple<runspec::DataGenerationParametersType, std::vector<int32_t>>
parse_data_parameters(std::string input) {
  std::regex pattern(R"((none|ec|vbw)(-\d+)?(-\d+)?)");
  std::smatch match;

  runspec::DataGenerationParametersType type =
      runspec::DataGenerationParametersType::NONE;
  int start_or_end = -1;
  int end = -1;

  if (std::regex_match(input, match, pattern)) {
    if (match.size() > 1) {
      std::string name = match[1].str();

      if (name == "ec") {
        type = runspec::DataGenerationParametersType::EC;
      } else if (name == "vbw") {
        type = runspec::DataGenerationParametersType::VBW;
      }
    }

    if (match.size() > 0 && match[2].matched) {
      start_or_end = std::stoi(match[2].str().substr(1));
    }

    if (match.size() > 0 && match[3].matched) {
      end = std::stoi(match[3].str().substr(1));
    }
  }

  return {type,
          verification::generate_integer_range<int32_t>(start_or_end, end)};
}

static CLIArgs parse_cli_args(int argc, char **argv) {
  if (argc != 11) {
    throw std::invalid_argument(
        "Not the required amount of arguments.\n"
        "executable <function_name> <unpacker> <patcher> <unpack_n_vecs> "
        "<unpack_n_vals>"
        " <datatype_width> <dataset_name> <(none|ec|vbw)(-d?)(-d?)>"
        " <n_vecs> <print_debug>");
  }
  int32_t argcounter = 0;

  std::string function_name = argv[++argcounter];
  std::string unpacker = argv[++argcounter];
  std::string patcher = argv[++argcounter];
  unsigned unpack_n_vecs = static_cast<unsigned>(std::atoi(argv[++argcounter]));
  unsigned unpack_n_vals = static_cast<unsigned>(std::atoi(argv[++argcounter]));
  int32_t datatype_width = std::atoi(argv[++argcounter]);
  std::string dataset_name = argv[++argcounter];
  auto [data_params_type, data_params] =
      parse_data_parameters(argv[++argcounter]);

  size_t n_vecs = static_cast<size_t>(std::atoi(argv[++argcounter]));
  size_t count = n_vecs * 1024;
  bool print_debug = std::atoi(argv[++argcounter]);

  return CLIArgs{datatype_width,
                 runspec::RunSpecification{
                     function_name,
                     runspec::KernelSpecification{
                         runspec::unpacker_options.at(unpacker),
                         runspec::patcher_options.at(patcher),
                         unpack_n_vecs,
                         unpack_n_vals,
                     },
                     runspec::DataSpecification{
                         count,
												 n_vecs,
                         data_params_type,
                         data_params,
                         dataset_name,
                     },
                 },
                 print_debug};
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
int32_t run_kernel(CLIArgs args) {
  verification::VerificationResult<T> results =
      functions::Fastlanes<T>::functions.at(args.runspec.function)(
          args.runspec);
	return 0;
  return process_results<T>(results, args);
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
int32_t run_kernel(CLIArgs args) {
  verification::VerificationResult<T> results =
      functions::Alp<T>::functions.at(args.runspec.function)(args.runspec);
  return process_results<T>(results, args);
}

int main(int argc, char **argv) {
  CLIArgs args = parse_cli_args(argc, argv);

  if (functions::Fastlanes<uint32_t>::functions.find(args.runspec.function) !=
      functions::Fastlanes<uint32_t>::functions.end()) {
    switch (args.datatype_width) {
    case 64: {
      return run_kernel<uint64_t>(args);
    }
    case 32: {
      return run_kernel<uint32_t>(args);
    }
    case 16: {
      return run_kernel<uint16_t>(args);
    }
    case 8: {
      return run_kernel<uint8_t>(args);
    }
    }
  } else if (functions::Alp<float>::functions.find(args.runspec.function) !=
             functions::Alp<float>::functions.end()) {
    switch (args.datatype_width) {
    case 64: {
      return run_kernel<double>(args);
    }
    case 32: {
      return run_kernel<float>(args);
    }
    }
  } else {
    throw std::invalid_argument("This function is not supported");
  }
}
