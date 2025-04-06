#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/data.cuh"
#include "engine/verification.cuh"
#include "flsgpu/flsgpu-api.cuh"
#include "generated-bindings/kernel-bindings.cuh"

enum class DataType {
  U32,
  U64,
  F32,
  F64,
};

enum class Kernel {
  Decompress,
  Query,
};

enum class Print {
  PrintNothing,
  PrintDebug,
  PrintDebugExit0,
};

struct ProgramParameters {
  DataType data_type;
  Kernel kernel;
  uint32_t unpack_n_vecs;
  uint32_t unpack_n_vals;
  bindings::Unpacker unpacker;
  bindings::Patcher patcher;
  data::ValueRange<vbw_t> bit_width_range;
  data::ValueRange<uint16_t> ec_range;
  size_t n_values;
  Print print_option;
};

struct CLIArgs {
  std::string data_type;
  std::string kernel;
  uint32_t unpack_n_vecs;
  uint32_t unpack_n_vals;
  std::string patcher;
  std::string unpacker;
  vbw_t start_vbw;
  vbw_t end_vbw;
  uint16_t start_ec;
  uint16_t end_ec;
  size_t n_vecs;
  uint32_t print_debug;

  CLIArgs(const int argc, char **argv) {
    constexpr int32_t CORRECT_ARG_COUNT = 13;
    if (argc != CORRECT_ARG_COUNT) {
      std::invalid_argument("Wrong arg count.\n");
    }
    int32_t argcounter = 0;

    data_type = argv[++argcounter];
    kernel = argv[++argcounter];
    unpack_n_vecs = std::stoul(argv[++argcounter]);
    unpack_n_vals = std::stoul(argv[++argcounter]);
    unpacker = argv[++argcounter];
    patcher = argv[++argcounter];
    start_vbw = std::stoul(argv[++argcounter]);
    end_vbw = std::stoul(argv[++argcounter]);
    start_ec = std::stoul(argv[++argcounter]);
    end_ec = std::stoul(argv[++argcounter]);
    n_vecs = std::stoul(argv[++argcounter]);
    print_debug = std::stoul(argv[++argcounter]);
  }

  ProgramParameters parse() {
    return ProgramParameters{
        string_to_data_type(data_type),
        string_to_kernel(kernel),
        unpack_n_vecs,
        unpack_n_vals,
        string_to_unpacker(unpacker),
        string_to_patcher(patcher),
        data::ValueRange<vbw_t>(start_vbw, end_vbw),
        data::ValueRange<uint16_t>(start_ec, end_ec),
        n_vecs * consts::VALUES_PER_VECTOR,
        static_cast<Print>(print_debug),
    };
  }

private:
  DataType string_to_data_type(const std::string &str) {
    static const std::unordered_map<std::string, DataType> mapping = {
        {"u32", DataType::U32},
        {"u64", DataType::U64},
        {"f32", DataType::F32},
        {"f64", DataType::F64},
    };

    auto it = mapping.find(str);
    if (it != mapping.end()) {
      return it->second;
    }

    throw std::invalid_argument("Unknown kernel type: " + str);
  }

  Kernel string_to_kernel(const std::string &str) {
    static const std::unordered_map<std::string, Kernel> mapping = {
        {"decompress", Kernel::Decompress},
        {"query", Kernel::Query},
    };

    auto it = mapping.find(str);
    if (it != mapping.end()) {
      return it->second;
    }

    throw std::invalid_argument("Unknown kernel type: " + str);
  }

  bindings::Unpacker string_to_unpacker(const std::string &str) {
    static const std::unordered_map<std::string, bindings::Unpacker> mapping = {
        {"stateless", bindings::Unpacker::Stateless},
        {"stateless-branchless", bindings::Unpacker::StatelessBranchless},
        {"stateful-cache", bindings::Unpacker::StatefulCache},
        {"stateful-local-1", bindings::Unpacker::StatefulLocal1},
        {"stateful-local-2", bindings::Unpacker::StatefulLocal2},
        {"stateful-local-4", bindings::Unpacker::StatefulLocal4},
        {"stateful-register-1", bindings::Unpacker::StatefulRegister1},
        {"stateful-register-2", bindings::Unpacker::StatefulRegister2},
        {"stateful-register-4", bindings::Unpacker::StatefulRegister4},
        {"stateful-register-branchless-1",
         bindings::Unpacker::StatefulRegisterBranchless1},
        {"stateful-register-branchless-2",
         bindings::Unpacker::StatefulRegisterBranchless2},
        {"stateful-register-branchless-4",
         bindings::Unpacker::StatefulRegisterBranchless4},
        {"stateful-branchless", bindings::Unpacker::StatefulBranchless},
    };

    auto it = mapping.find(str);
    if (it != mapping.end()) {
      return it->second;
    }

    throw std::invalid_argument("Unknown unpacker type: " + str);
  }

  bindings::Patcher string_to_patcher(const std::string &str) {
    static const std::unordered_map<std::string, bindings::Patcher> mapping = {
        {"none", bindings::Patcher::None},
        {"stateless", bindings::Patcher::Stateless},
        {"stateful", bindings::Patcher::Stateful},
        {"naive", bindings::Patcher::Naive},
        {"naive-branchless", bindings::Patcher::NaiveBranchless},
        {"prefetch-position", bindings::Patcher::PrefetchPosition},
        {"prefetch-all", bindings::Patcher::PrefetchAll},
        {"prefetch-all-branchless", bindings::Patcher::PrefetchAllBranchless},
    };

    auto it = mapping.find(str);
    if (it != mapping.end()) {
      return it->second;
    }

    throw std::invalid_argument("Unknown patcher type: " + str);
  }
};

template <typename T, typename ColumnT>
verification::ExecutionResult<T>
decompress_column(const ColumnT column, const ProgramParameters params) {
  auto column_device = column.copy_to_device();
  const T *out =
      bindings::decompress_column<T, typename ColumnT::DeviceColumnT>(
          column_device, params.unpack_n_vecs, params.unpack_n_vals,
          params.unpacker, params.patcher);
  flsgpu::host::free_column(column_device);

  const T *correct_out = data::bindings::decompress(column);
  auto result = verification::compare_data(correct_out, out, params.n_values);
  delete correct_out;
  return result;
}

template <typename T, typename ColumnT>
verification::ExecutionResult<T>
query_column(const ColumnT column, const ProgramParameters params,
             const bool query_result, const T magic_value) {
  auto column_device = column.copy_to_device();
  const bool answer =
      bindings::query_column<T, typename ColumnT::DeviceColumnT>(
          column_device, params.unpack_n_vecs, params.unpack_n_vals,
          params.unpacker, params.patcher, magic_value);
  flsgpu::host::free_column(column_device);

  // Weird hack to circumvent refactor_
  auto a = static_cast<T>(query_result);
  auto b = static_cast<T>(answer);

  return verification::compare_data(&a, &b, 1);
}

template <typename T, typename ColumnT>
verification::ExecutionResult<T>
execute_kernel(const ColumnT column, const ProgramParameters params,
               const bool query_result, const T magic_value) {
  if (params.kernel == Kernel::Decompress) {
    return decompress_column<T, ColumnT>(column, params);
  } else if (params.kernel == Kernel::Query) {
    return query_column<T, ColumnT>(column, params, query_result, magic_value);
  } else {
    throw std::invalid_argument("Kernel not implemented yet.\n");
  }
}

template <typename T>
std::vector<verification::ExecutionResult<T>>
execute_ffor(const ProgramParameters params) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  auto results = std::vector<verification::ExecutionResult<T>>();

  for (vbw_t vbw{params.bit_width_range.min}; vbw <= params.bit_width_range.max;
       ++vbw) {
    bool query_result = false;
    T magic_value = consts::as<T>::MAGIC_NUMBER;
    flsgpu::host::FFORColumn<T> column;

    if (params.kernel == Kernel::Query) {
      auto [_query_result, _column] =
          data::columns::generate_binary_ffor_column<T>(
              params.n_values, data::ValueRange<vbw_t>(vbw),
              params.unpack_n_vecs);
			query_result = _query_result;
			column = _column;
    } else {
      column = data::columns::generate_random_ffor_column<T>(
          params.n_values, data::ValueRange<vbw_t>(0, vbw),
          data::ValueRange<T>(0, 100), params.unpack_n_vecs);
    }

    results.push_back(execute_kernel<T, flsgpu::host::FFORColumn<T>>(
        column, params, query_result, magic_value));

    flsgpu::host::free_column(column);
  }

  return results;
}

template <typename T>
std::vector<verification::ExecutionResult<T>>
execute_alp(const ProgramParameters params) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  auto results = std::vector<verification::ExecutionResult<T>>();

  for (vbw_t vbw{params.bit_width_range.min}; vbw <= params.bit_width_range.max;
       ++vbw) {
    bool query_result = false;
    T magic_value = consts::as<T>::MAGIC_NUMBER;

    auto column = data::columns::generate_alp_column<T>(
        params.n_values, data::ValueRange<vbw_t>(0, vbw),
        data::ValueRange<uint16_t>(0), params.unpack_n_vecs);
    for (uint16_t ec{params.ec_range.min}; ec <= params.ec_range.max; ec += 10) {
      column = data::columns::modify_alp_exception_count(column, ec);

      if (params.kernel == Kernel::Query) {
        auto [_query_result, _magic_value] =
            data::columns::get_value_to_query<T, flsgpu::host::ALPColumn<T>>(
                column);
				query_result = _query_result;
				magic_value = _magic_value;
      }

      if (params.patcher == bindings::Patcher::Stateless ||
          params.patcher == bindings::Patcher::Stateful) {
        results.push_back(execute_kernel<T, flsgpu::host::ALPColumn<T>>(
            column, params, query_result, magic_value));
      } else {
        auto column_extended = column.create_extended_column();

        results.push_back(execute_kernel<T, flsgpu::host::ALPExtendedColumn<T>>(
            column_extended, params, query_result, magic_value));

        flsgpu::host::free_column(column_extended);
      }
    }

    flsgpu::host::free_column(column);
  }

  return results;
}

int main(int argc, char **argv) {
  CLIArgs args(argc, argv);
  ProgramParameters params = args.parse();

  int32_t exit_code = 0;
  bool print_debug = params.print_option != Print::PrintNothing;
  switch (params.data_type) {
  case DataType::U32:
    exit_code = verification::process_results(execute_ffor<uint32_t>(params),
                                              print_debug);
    break;
  case DataType::U64:
    exit_code = verification::process_results(execute_ffor<uint64_t>(params),
                                              print_debug);
    break;
  case DataType::F32:
    exit_code =
        verification::process_results(execute_alp<float>(params), print_debug);
    break;
  case DataType::F64:
    exit_code =
        verification::process_results(execute_alp<double>(params), print_debug);
    break;
  }

  if (params.print_option == Print::PrintDebugExit0) {
    exit(0);
  }

  exit(exit_code);
}
