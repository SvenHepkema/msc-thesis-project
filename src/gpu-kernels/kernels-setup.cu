#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>

#include "../alp/alp-bindings.hpp"
#include "../common/consts.hpp"
#include "../common/runspec.hpp"
#include "alp.cuh"
#include "host-alp-utils.cuh"
#include "host-utils.cuh"
#include "kernels-global.cuh"

namespace kernels {

template <typename T> struct ThreadblockMapping {
  static constexpr unsigned N_WARPS_PER_BLOCK =
      std::max(utils::get_n_lanes<T>() / consts::THREADS_PER_WARP, 2);
  static constexpr unsigned N_THREADS_PER_BLOCK =
      N_WARPS_PER_BLOCK * consts::THREADS_PER_WARP;
  static constexpr unsigned N_CONCURRENT_VECTORS_PER_BLOCK =
      N_THREADS_PER_BLOCK / utils::get_n_lanes<T>();

  const unsigned n_blocks;

  ThreadblockMapping(const runspec::KernelSpecification spec,
                     const size_t n_vecs)
      : n_blocks(n_vecs / (spec.n_vectors * N_CONCURRENT_VECTORS_PER_BLOCK)) {}
};

namespace fls {

#define FLS_DC(CASE, UNPACKER_T, N_VEC, N_VAL)                                 \
  case CASE: {                                                                 \
    kernels::device::fls::decompress_column<                                   \
        T, N_VEC, N_VAL, UNPACKER_T<T, N_VEC, N_VAL, BPFunctor<T>>>            \
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(                   \
            device_out.get(), device_in.get(), value_bit_width);               \
  } break;

template <typename T>
void verify_decompress_column(const runspec::KernelSpecification spec,
                              const T *__restrict in, T *__restrict out,
                              const size_t count,
                              const int32_t value_bit_width) {}

template <>
void verify_decompress_column<uint32_t>(const runspec::KernelSpecification spec,
                                        const uint32_t *__restrict in,
                                        uint32_t *__restrict out,
                                        const size_t count,
                                        const int32_t value_bit_width) {
  using T = uint32_t;
  const ThreadblockMapping<T> mapping(spec, utils::get_n_vecs_from_size(count));
  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  // The branchless version always does 1 access too many for each lane
  // That is why we allocate a little extra memory
  const size_t branchless_extra_access_buffer =
      sizeof(T) * utils::get_n_lanes<T>();
  GPUArray<T> device_in(encoded_count + branchless_extra_access_buffer, in);
  GPUArray<T> device_out(count);

  switch (spec.kernel) {
    FLS_DC(runspec::KernelOption::STATELESS_1_1, BitUnpackerStateless, 1, 1)
    FLS_DC(runspec::KernelOption::STATELESS_4_1, BitUnpackerStateless, 4, 1)
    FLS_DC(runspec::KernelOption::STATELESS_1_4, BitUnpackerStateless, 1, 4)
    FLS_DC(runspec::KernelOption::STATEFUL_1_1, BitUnpackerStateful, 1, 1)
    FLS_DC(runspec::KernelOption::STATEFUL_4_1, BitUnpackerStateful, 4, 1)
    FLS_DC(runspec::KernelOption::STATEFUL_1_4, BitUnpackerStateful, 1, 4)
    FLS_DC(runspec::KernelOption::STATELESS_BRANCHLESS_1_1,
           BitUnpackerStatelessBranchless, 1, 1)
    FLS_DC(runspec::KernelOption::STATELESS_BRANCHLESS_4_1,
           BitUnpackerStatelessBranchless, 4, 1)
    FLS_DC(runspec::KernelOption::STATELESS_BRANCHLESS_1_4,
           BitUnpackerStatelessBranchless, 1, 4)
    FLS_DC(runspec::KernelOption::STATEFUL_BRANCHLESS_1_1,
           BitUnpackerStatefulBranchless, 1, 1)
    FLS_DC(runspec::KernelOption::STATEFUL_BRANCHLESS_4_1,
           BitUnpackerStatefulBranchless, 4, 1)
    FLS_DC(runspec::KernelOption::STATEFUL_BRANCHLESS_1_4,
           BitUnpackerStatefulBranchless, 1, 4)
  default: {
    throw std::invalid_argument("Did not find this spec");
  } break;
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);
}

#define FLS_QCCZ(CASE, UNPACKER_T, N_VEC, N_VAL)                               \
  case CASE: {                                                                 \
    kernels::device::fls::query_column_contains_zero<                          \
        T, N_VEC, N_VAL, UNPACKER_T<T, N_VEC, N_VAL, BPFunctor<T>>>            \
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(                   \
            device_out.get(), device_in.get(), value_bit_width);               \
  } break;

template <typename T>
void query_column_contains_zero(const runspec::KernelSpecification spec,
                                const T *__restrict in, T *__restrict out,
                                const size_t count,
                                const int32_t value_bit_width) {}

template <>
void query_column_contains_zero<uint32_t>(
    const runspec::KernelSpecification spec, const uint32_t *__restrict in,
    uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width) {
  using T = uint32_t;
  const ThreadblockMapping<T> mapping(spec, utils::get_n_vecs_from_size(count));

  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  // The branchless version always does 1 access too many for each lane
  // That is why we allocate a little extra memory
  const size_t branchless_extra_access_buffer =
      sizeof(T) * utils::get_n_lanes<T>();
  GPUArray<T> device_in(encoded_count + branchless_extra_access_buffer, in);
  GPUArray<T> device_out(1);

  switch (spec.kernel) {
    FLS_QCCZ(runspec::KernelOption::STATELESS_1_1, BitUnpackerStateless, 1, 1)
    FLS_QCCZ(runspec::KernelOption::STATELESS_4_1, BitUnpackerStateless, 4, 1)
    FLS_QCCZ(runspec::KernelOption::STATELESS_1_4, BitUnpackerStateless, 1, 4)
    FLS_QCCZ(runspec::KernelOption::STATEFUL_1_1, BitUnpackerStateful, 1, 1)
    FLS_QCCZ(runspec::KernelOption::STATEFUL_4_1, BitUnpackerStateful, 4, 1)
    FLS_QCCZ(runspec::KernelOption::STATEFUL_1_4, BitUnpackerStateful, 1, 4)
    FLS_QCCZ(runspec::KernelOption::STATELESS_BRANCHLESS_1_1,
             BitUnpackerStatelessBranchless, 1, 1)
    FLS_QCCZ(runspec::KernelOption::STATELESS_BRANCHLESS_4_1,
             BitUnpackerStatelessBranchless, 4, 1)
    FLS_QCCZ(runspec::KernelOption::STATELESS_BRANCHLESS_1_4,
             BitUnpackerStatelessBranchless, 1, 4)
    FLS_QCCZ(runspec::KernelOption::STATEFUL_BRANCHLESS_1_1,
             BitUnpackerStatefulBranchless, 1, 1)
    FLS_QCCZ(runspec::KernelOption::STATEFUL_BRANCHLESS_4_1,
             BitUnpackerStatefulBranchless, 4, 1)
    FLS_QCCZ(runspec::KernelOption::STATEFUL_BRANCHLESS_1_4,
             BitUnpackerStatefulBranchless, 1, 4)
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

#define FLS_CC(CASE, UNPACKER_T, N_VEC, N_VAL)                                 \
  case CASE: {                                                                 \
    kernels::device::fls::compute_column<                                      \
        T, N_VEC, N_VAL, UNPACKER_T<T, N_VEC, N_VAL, BPFunctor<T>>,            \
        N_REPETITIONS><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(     \
        device_out.get(), device_in.get(), value_bit_width, runtime_zero);     \
  } break;

template <typename T>
void compute_column(const runspec::KernelSpecification spec,
                    const T *__restrict in, T *__restrict out,
                    const size_t count, const int32_t value_bit_width) {}

template <>
void compute_column<uint32_t>(const runspec::KernelSpecification spec,
                              const uint32_t *__restrict in,
                              uint32_t *__restrict out, const size_t count,
                              const int32_t value_bit_width) {
  using T = uint32_t;
  const ThreadblockMapping<T> mapping(spec, utils::get_n_vecs_from_size(count));

  const auto encoded_count =
      value_bit_width == 0
          ? 1
          : (count * static_cast<size_t>(value_bit_width)) / (8 * sizeof(T));

  // The branchless version always does 1 access too many for each lane
  // That is why we allocate a little extra memory
  const size_t branchless_extra_access_buffer =
      sizeof(T) * utils::get_n_lanes<T>();
  GPUArray<T> device_in(encoded_count + branchless_extra_access_buffer, in);
  GPUArray<T> device_out(1);
  uint32_t runtime_zero = 0;

  constexpr int N_REPETITIONS = 10;
  switch (spec.kernel) {
    FLS_CC(runspec::KernelOption::STATELESS_1_1, BitUnpackerStateless, 1, 1)
    FLS_CC(runspec::KernelOption::STATELESS_4_1, BitUnpackerStateless, 4, 1)
    FLS_CC(runspec::KernelOption::STATELESS_1_4, BitUnpackerStateless, 1, 4)
    FLS_CC(runspec::KernelOption::STATEFUL_1_1, BitUnpackerStateful, 1, 1)
    FLS_CC(runspec::KernelOption::STATEFUL_4_1, BitUnpackerStateful, 4, 1)
    FLS_CC(runspec::KernelOption::STATEFUL_1_4, BitUnpackerStateful, 1, 4)
    FLS_CC(runspec::KernelOption::STATELESS_BRANCHLESS_1_1,
           BitUnpackerStatelessBranchless, 1, 1)
    FLS_CC(runspec::KernelOption::STATELESS_BRANCHLESS_4_1,
           BitUnpackerStatelessBranchless, 4, 1)
    FLS_CC(runspec::KernelOption::STATELESS_BRANCHLESS_1_4,
           BitUnpackerStatelessBranchless, 1, 4)
    FLS_CC(runspec::KernelOption::STATEFUL_BRANCHLESS_1_1,
           BitUnpackerStatefulBranchless, 1, 1)
    FLS_CC(runspec::KernelOption::STATEFUL_BRANCHLESS_4_1,
           BitUnpackerStatefulBranchless, 4, 1)
    FLS_CC(runspec::KernelOption::STATEFUL_BRANCHLESS_1_4,
           BitUnpackerStatefulBranchless, 1, 4)
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

} // namespace fls

namespace gpualp {

#define ALP_DC(CASE, UNPACKER_T, PATCHER_T, N_VEC, N_VAL)                      \
  case CASE: {                                                                 \
    device_column = transfer::copy_alp_column_to_gpu(data);                    \
    kernels::device::alp::decompress_column<                                   \
        T, N_VEC, N_VAL,                                                       \
        AlpUnpacker<T, N_VEC, N_VAL,                                           \
                    UNPACKER_T<T, N_VEC, N_VAL, ALPFunctor<T>>,                \
                    PATCHER_T<T, N_VEC, N_VAL>, AlpColumn<T>>,                 \
        AlpColumn<T>><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(      \
        d_out.get(), device_column);                                           \
  } break;

#define ALP_DCE(CASE, UNPACKER_T, PATCHER_T, N_VEC, N_VAL)                     \
  case CASE: {                                                                 \
    device_extended_column = transfer::copy_alp_extended_column_to_gpu(data);  \
    kernels::device::alp::decompress_column<                                   \
        T, N_VEC, N_VAL,                                                       \
        AlpUnpacker<T, N_VEC, N_VAL,                                           \
                    UNPACKER_T<T, N_VEC, N_VAL, ALPFunctor<T>>,                \
                    PATCHER_T<T, N_VEC, N_VAL>, AlpExtendedColumn<T>>,         \
        AlpExtendedColumn<T>>                                                  \
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(                   \
            d_out.get(), device_extended_column);                              \
  } break;

template <typename T>
void verify_decompress_column(const runspec::KernelSpecification spec,
                              T *__restrict out,
                              const alp::AlpCompressionData<T> *data) {
  const auto count = data->size;
  const ThreadblockMapping<T> mapping(spec, utils::get_n_vecs_from_size(count));

  GPUArray<T> d_out(count);
  constant_memory::load_alp_constants<T>();

  AlpColumn<T> device_column;
  AlpExtendedColumn<T> device_extended_column;

  switch (spec.kernel) {
    ALP_DC(runspec::KernelOption::STATELESS_1_1, BitUnpackerStateless,
           StatelessALPExceptionPatcher, 1, 1)
    ALP_DCE(runspec::KernelOption::STATEFUL_1_1, BitUnpackerStateless,
            PrefetchAllALPExceptionPatcher, 1, 1)
  default: {
    throw std::invalid_argument("Did not find this spec");
  } break;
  }

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);

  transfer::destroy_alp_column(device_column);
  transfer::destroy_alp_column(device_extended_column);
}

#define ALP_QCCM(CASE, UNPACKER_T, PATCHER_T, N_VEC, N_VAL)                    \
  case CASE: {                                                                 \
    device_column = transfer::copy_alp_column_to_gpu(data);                    \
    kernels::device::alp::query_column_contains_magic<                         \
        T, N_VEC, N_VAL,                                                       \
        AlpUnpacker<T, N_VEC, N_VAL,                                           \
                    UNPACKER_T<T, N_VEC, N_VAL, ALPFunctor<T>>,                \
                    PATCHER_T<T, N_VEC, N_VAL>, AlpColumn<T>>,                 \
        AlpColumn<T>><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(      \
        d_out.get(), device_column, magic_value);                              \
  } break;

#define ALP_QCCME(CASE, UNPACKER_T, PATCHER_T, N_VEC, N_VAL)                   \
  case CASE: {                                                                 \
    device_extended_column = transfer::copy_alp_extended_column_to_gpu(data);  \
    kernels::device::alp::query_column_contains_magic<                         \
        T, N_VEC, N_VAL,                                                       \
        AlpUnpacker<T, N_VEC, N_VAL,                                           \
                    UNPACKER_T<T, N_VEC, N_VAL, ALPFunctor<T>>,                \
                    PATCHER_T<T, N_VEC, N_VAL>, AlpExtendedColumn<T>>,         \
        AlpExtendedColumn<T>>                                                  \
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(                   \
            d_out.get(), device_extended_column, magic_value);                 \
  } break;

template <typename T>
void query_column_contains_magic(const runspec::KernelSpecification spec,
                                 T *__restrict out,
                                 const alp::AlpCompressionData<T> *data,
                                 const T magic_value) {
  const auto count = data->size;
  const ThreadblockMapping<T> mapping(spec, utils::get_n_vecs_from_size(count));

  GPUArray<T> d_out(1);
  constant_memory::load_alp_constants<T>();

  AlpColumn<T> device_column;
  AlpExtendedColumn<T> device_extended_column;

  switch (spec.kernel) {
    ALP_QCCM(runspec::KernelOption::STATELESS_1_1, BitUnpackerStateless,
             StatelessALPExceptionPatcher, 1, 1)
    ALP_QCCME(runspec::KernelOption::STATEFUL_1_1, BitUnpackerStateless,
              PrefetchAllALPExceptionPatcher, 1, 1)
  default: {
    throw std::invalid_argument("Did not find this spec");
  } break;
  }

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);

  if (*out != static_cast<T>(true)) {
    *out = static_cast<T>(false);
  }

  transfer::destroy_alp_column(device_column);
  transfer::destroy_alp_column(device_extended_column);
}
} // namespace gpualp

} // namespace kernels

template void kernels::fls::verify_decompress_column<uint8_t>(
    const runspec::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
template void kernels::fls::verify_decompress_column<uint16_t>(
    const runspec::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void kernels::fls::verify_decompress_column<uint64_t>(
    const runspec::KernelSpecification spec, const uint64_t *__restrict in,
    uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

template void kernels::fls::query_column_contains_zero<uint8_t>(
    const runspec::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
template void kernels::fls::query_column_contains_zero<uint16_t>(
    const runspec::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void kernels::fls::query_column_contains_zero<uint64_t>(
    const runspec::KernelSpecification spec, const uint64_t *__restrict in,
    uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

template void kernels::fls::compute_column<uint8_t>(
    const runspec::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
template void kernels::fls::compute_column<uint16_t>(
    const runspec::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void kernels::fls::compute_column<uint64_t>(
    const runspec::KernelSpecification spec, const uint64_t *__restrict in,
    uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

template void kernels::gpualp::verify_decompress_column<float>(
    const runspec::KernelSpecification spec, float *__restrict out,
    const alp::AlpCompressionData<float> *data);
template void kernels::gpualp::verify_decompress_column<double>(
    const runspec::KernelSpecification spec, double *__restrict out,
    const alp::AlpCompressionData<double> *data);

template void kernels::gpualp::query_column_contains_magic<float>(
    const ::runspec::KernelSpecification spec, float *__restrict out,
    const alp::AlpCompressionData<float> *data, const float magic_value);
template void kernels::gpualp::query_column_contains_magic<double>(
    const ::runspec::KernelSpecification spec, double *__restrict out,
    const alp::AlpCompressionData<double> *data, const double magic_value);
