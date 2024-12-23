#include <cstddef>
#include <cstdint>

#ifndef GPU_FLS_HPP
#define GPU_FLS_HPP

namespace fls {
namespace gpu {
namespace test {

template <typename T, unsigned UNPACK_N_VECTORS = 1>
void bitunpack(const T *__restrict in, T *__restrict out, const size_t count,
               const int32_t value_bit_width);

template <typename T, unsigned UNPACK_N_VECTORS = 1>
void bitunpack_with_state(const T *__restrict in, T *__restrict out,
                          const size_t count, const int32_t value_bit_width);

template <typename T>
void unffor(const T *__restrict in, T *__restrict out, const size_t count,
            const int32_t value_bit_width, const T *__restrict base_p);

} // namespace test
} // namespace gpu
} // namespace fls

extern template void fls::gpu::test::bitunpack<uint8_t, 1>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void fls::gpu::test::bitunpack<uint16_t, 1>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void fls::gpu::test::bitunpack<uint32_t, 1>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void fls::gpu::test::bitunpack<uint64_t, 1>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

extern template void fls::gpu::test::bitunpack_with_state<uint8_t, 1>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void fls::gpu::test::bitunpack_with_state<uint16_t, 1>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void fls::gpu::test::bitunpack_with_state<uint32_t, 1>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void fls::gpu::test::bitunpack_with_state<uint64_t, 1>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

extern template void fls::gpu::test::bitunpack<uint8_t, 4>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void fls::gpu::test::bitunpack<uint16_t, 4>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void fls::gpu::test::bitunpack<uint32_t, 4>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width);
extern template void fls::gpu::test::bitunpack<uint64_t, 4>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width);

extern template void fls::gpu::test::unffor<uint8_t>(
    const uint8_t *__restrict in, uint8_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint8_t *__restrict base_p);
extern template void fls::gpu::test::unffor<uint16_t>(
    const uint16_t *__restrict in, uint16_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint16_t *__restrict base_p);
extern template void fls::gpu::test::unffor<uint32_t>(
    const uint32_t *__restrict in, uint32_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint32_t *__restrict base_p);
extern template void fls::gpu::test::unffor<uint64_t>(
    const uint64_t *__restrict in, uint64_t *__restrict out, const size_t count,
    const int32_t value_bit_width, const uint64_t *__restrict base_p);
#endif // GPU_FLS_HPP
