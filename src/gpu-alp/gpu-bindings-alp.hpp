#include <cstddef>
#include <cstdint>

#include "../alp/alp-bindings.hpp"

#ifndef GPU_ALP_HPP
#define GPU_ALP_HPP

namespace gpu {
template <typename T>
void alp(T *__restrict out, const alp::AlpCompressionData<T> *data);
} // namespace gpu

extern template void gpu::alp<float>(float *__restrict out,
                                      const alp::AlpCompressionData<float> *data);
extern template void gpu::alp<double>(double *__restrict out,
                                      const alp::AlpCompressionData<double> *data);
#endif // GPU_ALP_HPP
