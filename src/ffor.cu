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
    std::printf("Wrong arg count, correct usage: %s <n_values>\n", argv[0]);
    exit(1);
  }
  int32_t argcounter = 0;
  return CLIArgs{std::stoul(argv[++argcounter])};
}

int main(int argc, char **argv) {
  CLIArgs args = parse_cli_args(argc, argv);

  using T = uint32_t;

  auto failed_results = std::vector<verification::ExecutionResult<T>>();
  for (vbw_t i{0}; i < sizeof(T) * 8; ++i) {
    auto column = data::generators::columns::generate_random_bp_column<T>(
        args.n_values, data::generators::ValueRange<vbw_t>(0, sizeof(T) * 8));

    T *out_a = data::generators::fls_bindings::decompress(column);

    constexpr unsigned UNPACK_N_VECTORS = 1;
    constexpr unsigned UNPACK_N_VALUES = 1;
    T *out_b = kernels::host::decompress_column<
        T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
        flsgpu::device::BPDecompressor<
            T, flsgpu::device::BitUnpackerStatefulBranchless<
                   T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
                   flsgpu::device::BPFunctor<T>>>>(column);

    auto result = verification::compare_data(out_a, out_b, args.n_values);
    if (!result.success) {
      failed_results.push_back(result);
    }

    flsgpu::host::free_column(column);
		delete out_a;
		delete out_b;
  }

  for (auto fail : failed_results) {
    for (auto difference : fail.differences) {
      difference.log<T>();
    }
  }

  exit(failed_results.size());
}
