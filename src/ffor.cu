#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/data.cuh"
#include "engine/kernels.cuh"
#include "engine/verification.cuh"
#include "flsgpu/flsgpu-api.cuh"

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

int main(int argc, char **argv) {
  CLIArgs args = parse_cli_args(argc, argv);

  using T = double;
  using UINT_T = typename utils::same_width_uint<T>::type;

  auto results = std::vector<verification::ExecutionResult<T>>();

  for (vbw_t vbw{0}; vbw <= sizeof(T) * 8 / 2; ++vbw) {
    constexpr unsigned UNPACK_N_VECTORS = 1;
    constexpr unsigned UNPACK_N_VALUES = 1;

    auto column = data::columns::generate_alp_column<T>(
        args.n_values, data::ValueRange<vbw_t>(vbw),
        data::ValueRange<uint16_t>(20), UNPACK_N_VECTORS);
    // auto [array, size] =
    // data::arrays::read_file_as<T>("./data-input/basel_wind_f.csv.bin",
    // args.n_values);
    // auto column = alp::encode(array, size);
    auto out_a = alp::decode(column, new T[args.n_values]);

    T *out_b = kernels::host::decompress_column<
        T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
        flsgpu::device::ALPDecompressor<
            T, UNPACK_N_VECTORS,
            flsgpu::device::BitUnpackerStatefulBranchless<
                T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
                flsgpu::device::ALPFunctor<T, UNPACK_N_VECTORS>>,
            flsgpu::device::StatelessALPExceptionPatcher<T, UNPACK_N_VECTORS,
                                                         UNPACK_N_VALUES>,
            flsgpu::device::ALPColumn<T>>,
        flsgpu::host::ALPColumn<T>>(column);

    results.push_back(verification::compare_data(out_a, out_b, args.n_values));

    flsgpu::host::free_column(column);
    // delete[] array;
    delete[] out_a;
    delete[] out_b;
  }

  exit(verification::process_results(results, true));
}
