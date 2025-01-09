#include <cstdint>

#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "../gpu-alp/alp-benchmark-kernels-bindings.hpp"
#include "../gpu-fls/fls-benchmark-kernels-bindings.hpp"
#include "verification.hpp"

#ifndef QUERIES_H
#define QUERIES_H

namespace queries {
namespace baseline {
namespace cpu {
template <typename T> struct IntAnyValueIsMagicFn {
  void operator()(const T *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  const size_t a_count) {
    bool none_magic = 1;
    T *temp = new T[consts::VALUES_PER_VECTOR];
    auto n_vecs = utils::get_n_vecs_from_size(a_count);

    for (size_t i{0}; i < n_vecs; ++i) {
      for (size_t j{0}; j < consts::VALUES_PER_VECTOR; ++j) {
        temp[j] = a_in[i * consts::VALUES_PER_VECTOR + j];
      }

      for (size_t j{0}; j < consts::VALUES_PER_VECTOR; ++j) {
        none_magic &= temp[j] != consts::as<T>::MAGIC_NUMBER;
      }
    }

    *a_out = !none_magic;
    delete[] temp;
  }
};

template <typename T> struct FloatAnyValueIsMagicFn {
  void operator()(const T *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  const size_t a_count) {
    bool none_magic = true;
    for (size_t i{0}; i < a_count; ++i) {
      none_magic &= a_in[i] != consts::as<T>::MAGIC_NUMBER;
    }
    *a_out = static_cast<T>(!none_magic);
  }
};
} // namespace cpu

namespace gpu {
template <typename T> struct IntAnyValueIsMagicFn {
  void operator()(const T *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  const size_t a_count) {
    fls::gpu::bench::query_baseline_contains_zero<T>(a_in, a_out, a_count);
  }
};

template <typename T> struct FloatAnyValueIsMagicFn {
  void operator()(const T *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  const size_t a_count) {
    alp::gpu::bench::decode_baseline<T>(a_out, a_in, a_count);
  }
};
} // namespace gpu
} // namespace baseline

namespace FLS {
namespace cpu {
template <typename T> struct AnyValueIsMagicFn {
  void operator()(const T *a_in, T *a_out, const int32_t a_value_bit_width,
                  const size_t a_count) {
    bool none_magic = 1;
    T *temp = new T[1024];
    auto n_vecs = utils::get_n_vecs_from_size(a_count);
    size_t compressed_vector_size = static_cast<size_t>(
        utils::get_compressed_vector_size<T>(a_value_bit_width));

    for (size_t i{0}; i < n_vecs; ++i) {
      fls::unpack(a_in + i * compressed_vector_size, temp,
                  static_cast<uint8_t>(a_value_bit_width));

      for (size_t j{0}; j < 1024; ++j) {
        none_magic &= temp[j] != consts::as<T>::MAGIC_NUMBER;
      }
    }

    *a_out = !none_magic;
    delete[] temp;
  }
};
} // namespace cpu

namespace gpu {

template <typename T> struct OldAnyValueIsMagicFn {
  void operator()(const T *a_in, T *a_out, const int32_t a_value_bit_width,
                  const size_t a_count) {
    if (std::is_same<T, uint32_t>::value) {
      fls::gpu::bench::query_old_fls_contains_zero<uint32_t>(
          reinterpret_cast<const uint32_t *>(a_in),
          reinterpret_cast<uint32_t *>(a_out), a_count, a_value_bit_width);
    } else {
      throw std::invalid_argument("Invalid type.");
    }
  }
};

template <typename T> struct StatelessAnyValueIsMagicFn {
  void operator()(const T *a_in, T *a_out, const int32_t a_value_bit_width,
                  const size_t a_count) {
    fls::gpu::bench::query_bp_contains_zero<T>(a_in, a_out, a_count,
                                               a_value_bit_width, 1);
  }
};

template <typename T> struct StatefulAnyValueIsMagicFn {
  void operator()(const T *a_in, T *a_out, const int32_t a_value_bit_width,
                  const size_t a_count) {
    fls::gpu::bench::query_bp_stateful_contains_zero<T>(a_in, a_out, a_count,
                                                        a_value_bit_width, 1);
  }
};

} // namespace gpu
} // namespace FLS
namespace ALP {
namespace dynamic {
namespace cpu {

template <typename T> struct AnyValueIsMagicFn {
  void operator()(const alp::ALPMagicCompressionData<T> *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  const size_t a_count) {
    auto [data, magic_value] = (*a_in);
    T *temp = new T[a_count];
    alp::int_decode<T>(temp, data);

    bool none_magic = true;
    for (size_t i{0}; i < a_count; ++i) {
      none_magic &= temp[i] != magic_value;
    }
    *a_out = static_cast<T>(!none_magic);

    delete[] temp;
  }
};
} // namespace cpu

namespace gpu {

template <typename T> struct StatelessAnyValueIsMagicFn {
  void operator()(const alp::ALPMagicCompressionData<T> *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  [[maybe_unused]] const size_t a_count) {
    alp::gpu::bench::contains_magic_stateless<T>(a_out, a_in->first,
                                                 a_in->second);
  }
};

template <typename T> struct StatefulAnyValueIsMagicFn {
  void operator()(const alp::ALPMagicCompressionData<T> *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  [[maybe_unused]] const size_t a_count) {
    alp::gpu::bench::contains_magic_stateful<T>(a_out, a_in->first,
                                                a_in->second);
  }
};

template <typename T> struct StatefulExtendedAnyValueIsMagicFn {
  void operator()(const alp::ALPMagicCompressionData<T> *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  [[maybe_unused]] const size_t a_count) {
    alp::gpu::bench::contains_magic_stateful_extended<T, 1>(a_out, a_in->first,
                                                            a_in->second);
  }
};

} // namespace gpu
} // namespace dynamic

namespace constant {
namespace cpu {

template <typename T> struct AnyValueIsMagicFn {
  void operator()(const alp::AlpCompressionData<T> *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  const size_t a_count) {
    T *temp = new T[a_count];
    alp::int_decode<T>(temp, a_in);

    bool none_magic = true;
    for (size_t i{0}; i < a_count; ++i) {
      none_magic &= temp[i] != consts::as<T>::MAGIC_NUMBER;
    }
    *a_out = static_cast<T>(!none_magic);

    delete[] temp;
  }
};
} // namespace cpu

namespace gpu {

template <typename T> struct StatelessAnyValueIsMagicFn {
  void operator()(const alp::AlpCompressionData<T> *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  [[maybe_unused]] const size_t a_count) {
    alp::gpu::bench::contains_magic_stateless<T>(a_out, a_in,
                                                 consts::as<T>::MAGIC_NUMBER);
  }
};

template <typename T> struct StatefulAnyValueIsMagicFn {
  void operator()(const alp::AlpCompressionData<T> *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  [[maybe_unused]] const size_t a_count) {
    alp::gpu::bench::contains_magic_stateful<T>(a_out, a_in,
                                                consts::as<T>::MAGIC_NUMBER);
  }
};

template <typename T> struct StatefulExtendedAnyValueIsMagicFn {
  void operator()(const alp::AlpCompressionData<T> *a_in, T *a_out,
                  [[maybe_unused]] const int32_t a_value_bit_width,
                  [[maybe_unused]] const size_t a_count) {
    alp::gpu::bench::contains_magic_stateful_extended<T, 1>(
        a_out, a_in, consts::as<T>::MAGIC_NUMBER);
  }
};

} // namespace gpu
} // namespace constant
} // namespace ALP
} // namespace queries

#endif // QUERIES_H
