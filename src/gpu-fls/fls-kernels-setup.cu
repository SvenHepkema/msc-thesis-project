#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>

#include "../common/consts.hpp"
#include "../gpu-common/gpu-utils.cuh"
#include "fls-kernels-bindings.hpp"
#include "fls-kernels-global.cuh"

#define BU(CASE, UNPACKER_T, N_VEC, N_VAL)                                     \
  case CASE: {                                                                 \
    kernels::device::bitunpack<T, N_VEC, N_VAL,                                \
                               UNPACKER_T<T, N_VEC, N_VAL, BPFunctor<T>>>      \
        <<<n_blocks, n_threads>>>(device_out.get(), device_in.get(),           \
                                  value_bit_width);                            \
  } break;

#define QCCZ(CASE, UNPACKER_T, N_VEC, N_VAL)                                   \
  case CASE: {                                                                 \
    device::query_column_contains_zero<                                        \
        T, N_VEC, N_VAL, UNPACKER_T<T, N_VEC, N_VAL, BPFunctor<T>>>            \
        <<<n_blocks, n_threads>>>(device_out.get(), device_in.get(),           \
                                  value_bit_width);                            \
  } break;

namespace kernels {

template <>
void verify_bitunpack(const KernelSpecification spec,
                      const uint32_t *__restrict in, uint32_t *__restrict out,
                      const size_t count, const int32_t value_bit_width) {
  using T = uint32_t;
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  const auto n_threads = utils::get_n_lanes<T>();
  const auto n_blocks = n_vecs / spec.n_vectors;
  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(count);

  switch (spec.spec) {
    BU(TEST_STATELESS_1_1, BitUnpackerStateless, 1, 1);
    BU(TEST_STATEFUL_1_1, BitUnpackerStateful, 1, 1);
    BU(TEST_STATELESS_BRANCHLESS_1_1, BitUnpackerStatelessBranchless, 1, 1);
    BU(TEST_STATEFUL_BRANCHLESS_1_1, BitUnpackerStatefulBranchless, 1, 1);
  default: {
    throw std::invalid_argument("Did not find this spec");
  } break;
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);
}

// uint8_t, uint16_t, uint64_t
template <typename T>
void verify_bitunpack(const KernelSpecification spec, const T *__restrict in,
                      T *__restrict out, const size_t count,
                      const int32_t value_bit_width) {}

template <>
void query_column_contains_zero(const KernelSpecification spec,
                                const uint32_t *__restrict in,
                                uint32_t *__restrict out, const size_t count,
                                const int32_t value_bit_width) {
  using T = uint32_t;
  const auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  constexpr auto UNPACK_N_VECTORS = 1;
  const auto n_vectors_per_block = 2 * UNPACK_N_VECTORS;
  const auto n_blocks = n_vecs / n_vectors_per_block;
  const auto n_threads = utils::get_n_lanes<T>() * 2;

  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  GPUArray<T> device_in(encoded_count, in);
  GPUArray<T> device_out(1);

  switch (spec.spec) {
    QCCZ(QUERY_STATELESS_1_1, BitUnpackerStateless, 1, 1)
  default: {
    throw std::invalid_argument("Did not find this spec");
  } break;
  }

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);

  if (*out != 1) {
    *out = 0;
  }
}

template <typename T>
void query_column_contains_zero(const KernelSpecification spec,
                                const T *__restrict in, T *__restrict out,
                                const size_t count,
                                const int32_t value_bit_width) {}

} // namespace kernels

template void kernels::verify_bitunpack<uint8_t>(
    const kernels::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
template void kernels::verify_bitunpack<uint16_t>(
    const kernels::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void kernels::verify_bitunpack<uint64_t>(
    const kernels::KernelSpecification spec, const uint64_t *__restrict in,
    uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void kernels::query_column_contains_zero<uint8_t>(
    const kernels::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
template void kernels::query_column_contains_zero<uint16_t>(
    const kernels::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void kernels::query_column_contains_zero<uint64_t>(
    const kernels::KernelSpecification spec, const uint64_t *__restrict in,
    uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
