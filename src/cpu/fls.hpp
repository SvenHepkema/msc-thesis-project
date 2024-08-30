#include <cstdint>
#include <cstdio>
#include <functional>
#include <type_traits>

#include "../utils.hpp"

#ifndef CPU_FLS_HPP
#define CPU_FLS_HPP

namespace cpu {

template <typename T>
void bitpack(const T *__restrict in, T *__restrict out,
             [[maybe_unused]] const int32_t value_bit_width);
template <typename T>
void bitunpack(const T *__restrict in, T *__restrict out,
               [[maybe_unused]] const int32_t value_bit_width);
template <typename T>
void ffor(const T *__restrict in, T *__restrict out, const T *__restrict base_p,
          [[maybe_unused]] const int32_t value_bit_width);
template <typename T>
void unffor(const T *__restrict in, T *__restrict out,
            const T *__restrict base_p,
            [[maybe_unused]] const int32_t value_bit_width);
} // namespace cpu
extern template 
void cpu::bitpack(const uint8_t *__restrict in, uint8_t *__restrict out,
             [[maybe_unused]] const int32_t value_bit_width);
extern template
void cpu::bitunpack(const uint8_t *__restrict in, uint8_t *__restrict out,
               [[maybe_unused]] const int32_t value_bit_width);
extern template 
void cpu::ffor(const uint8_t *__restrict in, uint8_t *__restrict out, const uint8_t *__restrict base_p,
          [[maybe_unused]] const int32_t value_bit_width);
extern template
void cpu::unffor(const uint8_t *__restrict in, uint8_t *__restrict out,
            const uint8_t *__restrict base_p,
            [[maybe_unused]] const int32_t value_bit_width);

extern template 
void cpu::bitpack(const uint16_t *__restrict in, uint16_t *__restrict out,
             [[maybe_unused]] const int32_t value_bit_width);
extern template
void cpu::bitunpack(const uint16_t *__restrict in, uint16_t *__restrict out,
               [[maybe_unused]] const int32_t value_bit_width);
extern template 
void cpu::ffor(const uint16_t *__restrict in, uint16_t *__restrict out, const uint16_t *__restrict base_p,
          [[maybe_unused]] const int32_t value_bit_width);
extern template
void cpu::unffor(const uint16_t *__restrict in, uint16_t *__restrict out,
            const uint16_t *__restrict base_p,
            [[maybe_unused]] const int32_t value_bit_width);

extern template 
void cpu::bitpack(const uint32_t *__restrict in, uint32_t *__restrict out,
             [[maybe_unused]] const int32_t value_bit_width);
extern template
void cpu::bitunpack(const uint32_t *__restrict in, uint32_t *__restrict out,
               [[maybe_unused]] const int32_t value_bit_width);
extern template 
void cpu::ffor(const uint32_t *__restrict in, uint32_t *__restrict out, const uint32_t *__restrict base_p,
          [[maybe_unused]] const int32_t value_bit_width);
extern template
void cpu::unffor(const uint32_t *__restrict in, uint32_t *__restrict out,
            const uint32_t *__restrict base_p,
            [[maybe_unused]] const int32_t value_bit_width);

extern template 
void cpu::bitpack(const uint64_t *__restrict in, uint64_t *__restrict out,
             [[maybe_unused]] const int32_t value_bit_width);
extern template
void cpu::bitunpack(const uint64_t *__restrict in, uint64_t *__restrict out,
               [[maybe_unused]] const int32_t value_bit_width);
extern template 
void cpu::ffor(const uint64_t *__restrict in, uint64_t *__restrict out, const uint64_t *__restrict base_p,
          [[maybe_unused]] const int32_t value_bit_width);
extern template
void cpu::unffor(const uint64_t *__restrict in, uint64_t *__restrict out,
            const uint64_t *__restrict base_p,
            [[maybe_unused]] const int32_t value_bit_width);

#endif // CPU_FLS_HPP
