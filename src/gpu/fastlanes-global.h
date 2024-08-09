#include <cstddef>
#include <cstdint>

#ifndef FASTLANES_GLOBAL_H
#define FASTLANES_GLOBAL_H

namespace gpu {
template <typename T>
void bitunpack_with_function(const T *__restrict in, T *__restrict out,
                             const size_t count, const int32_t value_bit_width);
template <typename T>
void bitunpack_with_reader(const T *__restrict in, T *__restrict out,
                           const size_t count, const int32_t value_bit_width);
} // namespace gpu

extern template void
gpu::bitunpack_with_function<int8_t>(const int8_t *__restrict in,
                                     int8_t *__restrict out, const size_t count,
                                     const int32_t value_bit_width);
extern template void gpu::bitunpack_with_reader<int8_t>(const int8_t *__restrict in,
                                                 int8_t *__restrict out,
                                                 const size_t count,
                                                 const int32_t value_bit_width);
extern template void gpu::bitunpack_with_function<int16_t>(
    const int16_t *__restrict in, int16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void
gpu::bitunpack_with_reader<int16_t>(const int16_t *__restrict in,
                                    int16_t *__restrict out, const size_t count,
                                    const int32_t value_bit_width);
extern template void gpu::bitunpack_with_function<int32_t>(
    const int32_t *__restrict in, int32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void
gpu::bitunpack_with_reader<int32_t>(const int32_t *__restrict in,
                                    int32_t *__restrict out, const size_t count,
                                    const int32_t value_bit_width);
extern template void gpu::bitunpack_with_function<int64_t>(
    const int64_t *__restrict in, int64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void
gpu::bitunpack_with_reader<int64_t>(const int64_t *__restrict in,
                                    int64_t *__restrict out, const size_t count,
                                    const int32_t value_bit_width);
extern template void gpu::bitunpack_with_function<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void
gpu::bitunpack_with_reader<uint8_t>(const uint8_t *__restrict in,
                                    uint8_t *__restrict out, const size_t count,
                                    const int32_t value_bit_width);
extern template void gpu::bitunpack_with_function<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void gpu::bitunpack_with_reader<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void gpu::bitunpack_with_function<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void gpu::bitunpack_with_reader<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void gpu::bitunpack_with_function<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void gpu::bitunpack_with_reader<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
#endif // FASTLANES_GLOBAL_H
