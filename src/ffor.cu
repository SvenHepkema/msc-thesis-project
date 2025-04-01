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

  using T = uint32_t;

  auto results = std::vector<verification::ExecutionResult<T>>();

  for (vbw_t vbw{0}; vbw <= sizeof(T) * 8; ++vbw) {
    auto array =
        data::generators::arrays::generate_index_array<T>(args.n_values, vbw);

    auto column =
        data::generators::fls_bindings::compress(array, args.n_values, vbw);

    constexpr unsigned UNPACK_N_VECTORS = 1;
    constexpr unsigned UNPACK_N_VALUES = 1;
    T *out = kernels::host::decompress_column<
        T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
        flsgpu::device::BPDecompressor<
            T, flsgpu::device::BitUnpackerStatefulBranchless<
                   T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
                   flsgpu::device::BPFunctor<T>>>,
        flsgpu::host::BPColumn<T>>(column);

    results.push_back(verification::compare_data(array, out, args.n_values));

    flsgpu::host::free_column(column);
    delete[] array;
    delete[] out;
  }

  exit(verification::process_results(results, true));
}
