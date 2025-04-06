#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/data.cuh"
#include "engine/verification.cuh"
#include "flsgpu/flsgpu-api.cuh"
#include "generated-bindings/kernel-bindings.cuh"

struct ProgramParameters {
  enums::DataType data_type;
  enums::Kernel kernel;
  std::string file;
  data::ValueRange<vbw_t> bit_width_range;
  data::ValueRange<uint16_t> ec_range;
  size_t n_values;
  enums::Print print_option;

  bool read_data_from_file() const { return file != "generate"; }
};

struct CLIArgs {
  std::string data_type;
  std::string kernel;
  std::string file;
  vbw_t start_vbw;
  vbw_t end_vbw;
  uint16_t start_ec;
  uint16_t end_ec;
  size_t n_vecs;
  uint32_t print_debug;

  CLIArgs(const int argc, char **argv) {
    constexpr int32_t CORRECT_ARG_COUNT = 10;
    if (argc != CORRECT_ARG_COUNT) {
      throw std::invalid_argument("Wrong arg count.\n");
    }

    int32_t argcounter = 0;
    data_type = argv[++argcounter];
    kernel = argv[++argcounter];
    file = argv[++argcounter];
    start_vbw = std::stoul(argv[++argcounter]);
    end_vbw = std::stoul(argv[++argcounter]);
    start_ec = std::stoul(argv[++argcounter]);
    end_ec = std::stoul(argv[++argcounter]);
    n_vecs = std::stoul(argv[++argcounter]);
    print_debug = std::stoul(argv[++argcounter]);
  }

  ProgramParameters parse() {
    return ProgramParameters{
        enums::string_to_data_type(data_type),
        enums::string_to_kernel(kernel),
        file,
        data::ValueRange<vbw_t>(start_vbw, end_vbw),
        data::ValueRange<uint16_t>(start_ec, end_ec),
        n_vecs * consts::VALUES_PER_VECTOR,
        static_cast<enums::Print>(print_debug),
    };
  }
};

template <typename T>
void execute_ffor_decompress(flsgpu::device::FFORColumn<T> column_device) {
  const unsigned unpack_n_values = 1;
  const std::vector<unsigned> unpack_n_vecs_set = {1, 4};

  for (const auto unpack_n_vecs : unpack_n_vecs_set) {
    for (size_t u{0};
         u < static_cast<size_t>(enums::Unpacker::StatefulBranchless); ++u) {
      const auto unpacker = static_cast<enums::Unpacker>(u);
      const T *out =
          bindings::decompress_column<T, flsgpu::device::FFORColumn<T>>(
              column_device, unpack_n_vecs, unpack_n_values, unpacker,
              enums::Patcher::None);
      delete[] out;
    }
  }
}
template <typename T>
int32_t execute_ffor_query(flsgpu::device::FFORColumn<T> column_device,
                           const bool column_contains_value,
                           const T value_to_query) {
  const unsigned unpack_n_values = 1;
  const std::vector<unsigned> unpack_n_vecs_set = {1, 4};
  int32_t failed = 0;

  for (const auto unpack_n_vecs : unpack_n_vecs_set) {
    for (size_t u{0};
         u < static_cast<size_t>(enums::Unpacker::StatefulBranchless); ++u) {
      const auto unpacker = static_cast<enums::Unpacker>(u);
      const bool answer =
          bindings::query_column<T, flsgpu::device::FFORColumn<T>>(
              column_device, unpack_n_vecs, unpack_n_values, unpacker,
              enums::Patcher::None, consts::as<T>::MAGIC_NUMBER);
      failed += answer;
    }
  }

  return failed;
}

template <typename T> int32_t execute_ffor(const ProgramParameters params) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  auto results = std::vector<verification::ExecutionResult<T>>();

  int32_t failed = 0;

  for (vbw_t vbw{params.bit_width_range.min}; vbw <= params.bit_width_range.max;
       ++vbw) {
    if (params.kernel == enums::Kernel::Query) {
      auto [query_result, column] =
          data::columns::generate_binary_ffor_column<T>(
              params.n_values, data::ValueRange<vbw_t>(vbw),
              consts::MAX_UNPACK_N_VECS);
      auto column_device = column.copy_to_device();

      failed += execute_ffor_query(column_device, query_result,
                                   consts::as<T>::MAGIC_NUMBER);

      flsgpu::host::free_column(column_device);
      flsgpu::host::free_column(column);
    } else if (params.kernel == enums::Kernel::Decompress) {
      auto column = data::columns::generate_random_ffor_column<T>(
          params.n_values, data::ValueRange<vbw_t>(vbw),
          data::ValueRange<T>(0, 100), consts::MAX_UNPACK_N_VECS);
      auto column_device = column.copy_to_device();

      execute_ffor_decompress(column_device);

      flsgpu::host::free_column(column_device);
      flsgpu::host::free_column(column);
    }
  }

  return failed;
}

template <typename T> int32_t execute_alp(const ProgramParameters params) {
  return 0;
}

int main(int argc, char **argv) {
  CLIArgs args(argc, argv);
  ProgramParameters params = args.parse();

  int32_t exit_code = 0;
  bool print_debug = params.print_option != enums::Print::PrintNothing;
  switch (params.data_type) {
  case enums::DataType::U32:
    exit_code = execute_ffor<uint32_t>(params);

    break;
  case enums::DataType::U64:
    exit_code = execute_ffor<uint64_t>(params);
    break;
  case enums::DataType::F32:
    exit_code = execute_alp<float>(params);
    break;
  case enums::DataType::F64:
    exit_code = execute_alp<double>(params);
    break;
  }

	if (print_debug) {
		printf("Exit code: %d\n", exit_code);
	}

  if (params.print_option == enums::Print::PrintDebugExit0) {
    exit(0);
  }

  exit(exit_code);
}
