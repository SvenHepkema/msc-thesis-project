#include <cstddef>
#include <cstdint>

#ifndef GPU_FASTLANES_H
#define GPU_FASTLANES_H

namespace gpu {
template <typename T>
void bitunpack(const T *__restrict in, T *__restrict out,
                             const size_t count, const int32_t value_bit_width);
}

extern template void gpu::bitunpack<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void gpu::bitunpack<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void gpu::bitunpack<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void gpu::bitunpack<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
#endif // GPU_FASTLANES_H
