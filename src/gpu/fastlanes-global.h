#include <cstdint>
#include <cstddef>

#ifndef FASTLANES_GLOBAL_H
#define FASTLANES_GLOBAL_H

namespace gpu {
template <typename T_in, typename T_out>
void bitunpack_with_function(const T_in *__restrict in, T_out *__restrict out,
		const size_t count,
                             const int32_t value_bit_width) {
	return;
}
template <typename T_in, typename T_out>
void bitunpack_with_reader(const T_in *__restrict in, T_out *__restrict out,
		const size_t count,
                           const int32_t value_bit_width) {
	return;
}
} // namespace gpu

#endif // FASTLANES_GLOBAL_H
