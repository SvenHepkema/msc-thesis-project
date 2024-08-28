#include <cstddef>
#include <cstdint>

#ifndef GPU_FASTLANES_H
#define GPU_FASTLANES_H

namespace gpu {
template <typename T>
void bitunpack(const T *__restrict in, T *__restrict out, const size_t count,
               const int32_t value_bit_width);
template <typename T>
void unffor(const T *__restrict in, T *__restrict out, const size_t count,
            const int32_t value_bit_width, const T *__restrict base_p);
} // namespace gpu

extern template void gpu::bitunpack<uint8_t>(const uint8_t *__restrict in,
                                             uint8_t *__restrict out,
                                             const size_t count,
                                             const int32_t value_bit_width);
extern template void gpu::bitunpack<uint16_t>(const uint16_t *__restrict in,
                                              uint16_t *__restrict out,
                                              const size_t count,
                                              const int32_t value_bit_width);
extern template void gpu::bitunpack<uint32_t>(const uint32_t *__restrict in,
                                              uint32_t *__restrict out,
                                              const size_t count,
                                              const int32_t value_bit_width);
extern template void gpu::bitunpack<uint64_t>(const uint64_t *__restrict in,
                                              uint64_t *__restrict out,
                                              const size_t count,
                                              const int32_t value_bit_width);
extern template void gpu::unffor<uint8_t>(const uint8_t *__restrict in,
                                          uint8_t *__restrict out,
                                          const size_t count,
                                          const int32_t value_bit_width,
                                          const uint8_t *__restrict base_p);
extern template void gpu::unffor<uint16_t>(const uint16_t *__restrict in,
                                           uint16_t *__restrict out,
                                           const size_t count,
                                           const int32_t value_bit_width,
                                           const uint16_t *__restrict base_p);
extern template void gpu::unffor<uint32_t>(const uint32_t *__restrict in,
                                           uint32_t *__restrict out,
                                           const size_t count,
                                           const int32_t value_bit_width,
                                           const uint32_t *__restrict base_p);
extern template void gpu::unffor<uint64_t>(const uint64_t *__restrict in,
                                           uint64_t *__restrict out,
                                           const size_t count,
                                           const int32_t value_bit_width,
                                           const uint64_t *__restrict base_p);
#endif // GPU_FASTLANES_H
