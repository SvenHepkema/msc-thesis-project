#include <cstdint>

#ifndef FASTLANES_GLOBAL_H
#define FASTLANES_GLOBAL_H

namespace gpu {
template <typename T_in, typename T_out>
void bitunpack_with_function(T_in *__restrict in, T_out *__restrict out,
                             uint16_t value_bit_width) {
	return;
}
template <typename T_in, typename T_out>
void bitunpack_with_reader(T_in *__restrict in, T_out *__restrict out,
                           uint16_t value_bit_width) {
	return;
}
} // namespace gpu

#endif // FASTLANES_GLOBAL_H
