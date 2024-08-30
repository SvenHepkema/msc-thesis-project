#ifndef FFOR_FFOR_HPP
#define FFOR_FFOR_HPP

#include <cstdint>

namespace fls {

void pack(const uint64_t *__restrict in, uint64_t *__restrict out, uint8_t bw);
void pack(const uint32_t *__restrict in, uint32_t *__restrict out, uint8_t bw);
void pack(const uint16_t *__restrict in, uint16_t *__restrict out, uint8_t bw);
void pack(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t bw);

void unpack(const uint64_t *__restrict in, uint64_t *__restrict out,
            uint8_t bw);
void unpack(const uint32_t *__restrict in, uint32_t *__restrict out,
            uint8_t bw);
void unpack(const uint16_t *__restrict in, uint16_t *__restrict out,
            uint8_t bw);
void unpack(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t bw);

void ffor(const uint64_t *__restrict in, uint64_t *__restrict out, uint8_t bw,
          const uint64_t *__restrict a_base_p);
void ffor(const uint32_t *__restrict in, uint32_t *__restrict out, uint8_t bw,
          const uint32_t *__restrict a_base_p);
void ffor(const uint16_t *__restrict in, uint16_t *__restrict out, uint8_t bw,
          const uint16_t *__restrict a_base_p);
void ffor(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t bw,
          const uint8_t *__restrict a_base_p);

void unffor(const uint64_t *__restrict in, uint64_t *__restrict out, uint8_t bw,
            const uint64_t *__restrict a_base_p);
void unffor(const uint32_t *__restrict in, uint32_t *__restrict out, uint8_t bw,
            const uint32_t *__restrict a_base_p);
void unffor(const uint16_t *__restrict in, uint16_t *__restrict out, uint8_t bw,
            const uint16_t *__restrict a_base_p);
void unffor(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t bw,
            const uint8_t *__restrict a_base_p);

} // namespace fastlanes

#endif
