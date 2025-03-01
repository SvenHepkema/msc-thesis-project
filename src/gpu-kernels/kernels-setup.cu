#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>

#include "../alp/alp-bindings.hpp"
#include "../common/consts.hpp"
#include "../common/runspec.hpp"
#include "alp.cuh"
#include "generated-kernel-calls.cuh"
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
      : n_blocks(n_vecs / (spec.n_vecs * N_CONCURRENT_VECTORS_PER_BLOCK)) {}
};

namespace fls {

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
      sizeof(T) * utils::get_n_lanes<T>() * 4;
  GPUArray<T> device_in(encoded_count, branchless_extra_access_buffer, in);
  GPUArray<T> device_out(count);

  generated_kernel_calls::fls_decompress_column(
      spec, mapping.n_blocks, mapping.N_THREADS_PER_BLOCK, device_out.get(),
      device_in.get(), value_bit_width);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);
}

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
      sizeof(T) * utils::get_n_lanes<T>() * 4;
  GPUArray<T> device_in(encoded_count, branchless_extra_access_buffer, in);
  GPUArray<T> device_out(1);

  generated_kernel_calls::fls_query_column<T>(
      spec, mapping.n_blocks, mapping.N_THREADS_PER_BLOCK, device_out.get(),
      device_in.get(), value_bit_width);

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);

  if (*out != 1) {
    *out = 0;
  }
}

template <typename T>
void query_column_contains_zero_unrolled(
    const runspec::KernelSpecification spec, const T *__restrict in,
    T *__restrict out, const size_t count, const int32_t value_bit_width) {}

template <>
void query_column_contains_zero_unrolled<uint32_t>(
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
      sizeof(T) * utils::get_n_lanes<T>() * 4;
  GPUArray<T> device_in(encoded_count, branchless_extra_access_buffer, in);
  GPUArray<T> device_out(1);

  generated_kernel_calls::fls_query_column_unrolled<T>(
      spec, mapping.n_blocks, mapping.N_THREADS_PER_BLOCK, device_out.get(),
      device_in.get(), value_bit_width);

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);

  if (*out != 1) {
    *out = 0;
  }
}

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
      sizeof(T) * utils::get_n_lanes<T>() * 4;
  GPUArray<T> device_in(encoded_count, branchless_extra_access_buffer, in);
  GPUArray<T> device_out(1);

  generated_kernel_calls::fls_compute_column(
      spec, mapping.n_blocks, mapping.N_THREADS_PER_BLOCK, device_out.get(),
      device_in.get(), value_bit_width);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  device_out.copy_to_host(out);

  if (*out != 1) {
    *out = 0;
  }
}

} // namespace fls

namespace gpualp {

template <typename T>
void verify_decompress_column(const runspec::KernelSpecification spec,
                              T *__restrict out,
                              const alp::AlpCompressionData<T> *data) {
  const auto count = data->size;
  const ThreadblockMapping<T> mapping(spec, utils::get_n_vecs_from_size(count));

  GPUArray<T> d_out(count);
  constant_memory::load_alp_constants<T>();

  generated_kernel_calls::alp_decompress_column(
      spec, mapping.n_blocks, mapping.N_THREADS_PER_BLOCK, d_out.get(), data);

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);
}

template <typename T>
void query_column_contains_magic(const runspec::KernelSpecification spec,
                                 T *__restrict out,
                                 const alp::AlpCompressionData<T> *data,
                                 const T magic_value) {
  const auto count = data->size;
  const ThreadblockMapping<T> mapping(spec, utils::get_n_vecs_from_size(count));

  GPUArray<T> d_out(1);
  constant_memory::load_alp_constants<T>();

  generated_kernel_calls::alp_query_column(spec, mapping.n_blocks,
                                           mapping.N_THREADS_PER_BLOCK,
                                           d_out.get(), data, magic_value);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_out.copy_to_host(out);

  if (*out != static_cast<T>(true)) {
    *out = static_cast<T>(false);
  }
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

template void kernels::fls::query_column_contains_zero_unrolled<uint8_t>(
    const runspec::KernelSpecification spec, const uint8_t *__restrict in,
    uint8_t *__restrict out, const size_t count, const int32_t value_bit_width);
template void kernels::fls::query_column_contains_zero_unrolled<uint16_t>(
    const runspec::KernelSpecification spec, const uint16_t *__restrict in,
    uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void kernels::fls::query_column_contains_zero_unrolled<uint64_t>(
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
