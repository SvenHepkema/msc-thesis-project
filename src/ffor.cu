#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/data.cuh"
#include "engine/verification.cuh"
#include "flsgpu/flsgpu-api.cuh"
#include "generated-bindings/kernel-bindings.cuh"

struct CLIArgs {
  size_t n_values;
};

CLIArgs parse_cli_args(const int argc, char **argv) {
  if (argc != 2) {
    std::printf("Wrong arg count, correct usage: %s <n_vecs>\n", argv[0]);
    exit(1);
  }
  int32_t argcounter = 0;
  return CLIArgs{std::stoul(argv[++argcounter]) * consts::VALUES_PER_VECTOR};
}

template <typename T>
std::vector<verification::ExecutionResult<T>> test_alp(CLIArgs args) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  auto results = std::vector<verification::ExecutionResult<T>>();

  for (vbw_t vbw{0}; vbw <= sizeof(T) * 8 / 2; ++vbw) {
    constexpr unsigned unpack_n_vectors = 4;
    constexpr unsigned unpack_n_values = 1;

    auto column = data::columns::generate_alp_column<T>(
        args.n_values, data::ValueRange<vbw_t>(vbw),
        data::ValueRange<uint16_t>(20), unpack_n_vectors);
    // auto [array, size] =
    // data::arrays::read_file_as<T>("./data-input/basel_wind_f.csv.bin",
    // args.n_values);
    // auto column = alp::encode(array, size);
    auto out_a = alp::decode(column, new T[args.n_values]);

    auto column_extended = column.create_extended_column();
    auto column_extended_device = column_extended.copy_to_device();
    T *out_b =
        bindings::decompress_column<T, flsgpu::device::ALPExtendedColumn<T>>(
            column_extended_device, unpack_n_vectors, unpack_n_values,
            bindings::Unpacker::StatefulBranchless,
            bindings::Patcher::NaiveBranchless);

    results.push_back(verification::compare_data(out_a, out_b, args.n_values));

    flsgpu::host::free_column(column_extended_device);
    flsgpu::host::free_column(column_extended);
    flsgpu::host::free_column(column);
    delete[] out_a;
    delete[] out_b;
  }

  return results;
}

int main(int argc, char **argv) {
  CLIArgs args = parse_cli_args(argc, argv);

  // auto results = test_ffor<uint32_t>(args);
  auto results = test_alp<float>(args);
  exit(verification::process_results(results, true));
}
