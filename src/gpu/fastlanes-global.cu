#include "../consts.h"
#include "fastlanes-global.cuh"
#include "fastlanes-global.h"
#include <cstddef>
#include <cstdint>

template <typename T>
void gpu::bitunpack_with_function(const T *__restrict in, T *__restrict out,
                                  const size_t count,
                                  const int32_t value_bit_width) {
  auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  bitunpack_with_function_global<T, T, 1, utils::get_values_per_lane<T>()>
      <<<n_vecs, 32>>>(in, out, value_bit_width);
}
template <typename T>
void gpu::bitunpack_with_reader(const T *__restrict in, T *__restrict out,
                                const size_t count,
                                const int32_t value_bit_width) {
  auto n_vecs = static_cast<uint32_t>(count / consts::VALUES_PER_VECTOR);
  bitunpack_with_reader_global<T, T, 1, utils::get_values_per_lane<T>()>
      <<<n_vecs, 32>>>(in, out, value_bit_width);
}

template void
gpu::bitunpack_with_function<int8_t>(const int8_t *__restrict in,
                                     int8_t *__restrict out, const size_t count,
                                     const int32_t value_bit_width);
template void gpu::bitunpack_with_reader<int8_t>(const int8_t *__restrict in,
                                                 int8_t *__restrict out,
                                                 const size_t count,
                                                 const int32_t value_bit_width);
template void gpu::bitunpack_with_function<int16_t>(
    const int16_t *__restrict in, int16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void
gpu::bitunpack_with_reader<int16_t>(const int16_t *__restrict in,
                                    int16_t *__restrict out, const size_t count,
                                    const int32_t value_bit_width);
template void gpu::bitunpack_with_function<int32_t>(
    const int32_t *__restrict in, int32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void
gpu::bitunpack_with_reader<int32_t>(const int32_t *__restrict in,
                                    int32_t *__restrict out, const size_t count,
                                    const int32_t value_bit_width);
template void gpu::bitunpack_with_function<int64_t>(
    const int64_t *__restrict in, int64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void
gpu::bitunpack_with_reader<int64_t>(const int64_t *__restrict in,
                                    int64_t *__restrict out, const size_t count,
                                    const int32_t value_bit_width);
template void gpu::bitunpack_with_function<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void
gpu::bitunpack_with_reader<uint8_t>(const uint8_t *__restrict in,
                                    uint8_t *__restrict out, const size_t count,
                                    const int32_t value_bit_width);
template void gpu::bitunpack_with_function<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void gpu::bitunpack_with_reader<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void gpu::bitunpack_with_function<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void gpu::bitunpack_with_reader<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void gpu::bitunpack_with_function<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
template void gpu::bitunpack_with_reader<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
