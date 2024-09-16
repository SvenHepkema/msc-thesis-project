#include <cstddef>
#include <cstdint>
#include <exception>
#include <type_traits>

#include "../common/utils.hpp"
// WARNING The original ALP repo contains code that triggers warnings if all
// warnings are turned off. To make sure these warnings do not show up when the
// alp directory itself is not recompiled, I added this pragma to show it as a
// system header. So be carefult, warnings from the alp/* files do not show up
// when compiling
#pragma clang system_header

#ifndef ALP_BINDINGS_HPP
#define ALP_BINDINGS_HPP

#include "config.hpp"

namespace alp {

class EncodingException : public std::exception {
public:
  using std::exception::what;
  char *what() { return "Could not encode data with desired encoding."; }
};

template <typename T> struct AlpFFORVecHeader {
  T *array;
  T *base;
  uint8_t *bit_width;
};

template <typename T> struct AlpFFORArray {
  size_t count;

  T *array;
  T *bases;
  uint8_t *bit_widths; // WARNING TODO I don't think the bit width changes for
                       // ALPRD, so only need singular

  AlpFFORArray<T>(const size_t count) {
    const size_t n_vecs = utils::get_n_vecs_from_size(count);

    // TODO: This should not allocate the maximum amount of space
    array = new T[count];
    bases = new T[n_vecs];
    bit_widths = new uint8_t[n_vecs];
  }

  AlpFFORVecHeader<T> get_ffor_header_for_vec(const int64_t vec_index) {
    return AlpFFORVecHeader<T>{
        array + vec_index * 1024,
        bases + vec_index,
        bit_widths + vec_index,
    };
  }

  ~AlpFFORArray<T>() {
    delete[] array;
    delete[] bases;
    delete[] bit_widths;
  }
};

template <typename T> struct AlpVecExceptions {
  T *exceptions;
  uint16_t *positions;
  uint16_t *count;
};

template <typename T> struct AlpExceptions {
  T *exceptions;
  uint16_t *counts;
  uint16_t *positions;

  AlpExceptions<T>(const size_t size) {
    const size_t n_vecs = utils::get_n_vecs_from_size(size);
    // TODO: This should not allocate the maximum number of exceptions
    exceptions = new T[size];
    positions = new uint16_t[size];
    counts = new uint16_t[n_vecs];
  }

  AlpVecExceptions<T> get_exceptions_for_vec(const int64_t vec_index) {
    return AlpVecExceptions<T>{
        exceptions + vec_index * 1024,
        positions + vec_index * 1024,
        counts + vec_index,
    };
  }

  ~AlpExceptions<T>() {
    delete[] exceptions;
    delete[] positions;
    delete[] counts;
  }
};

template <typename T> struct AlpCompressionData {
  using UINT_T = typename utils::same_width_uint<T>::type;

  size_t rowgroup_offset = 0;
  size_t size;

  AlpFFORArray<UINT_T> ffor;
  uint8_t *exponents;
  uint8_t *factors;

  AlpExceptions<T> exceptions;

  AlpCompressionData<T>(const size_t size_a)
      : size(size_a), ffor(size_a), exceptions(AlpExceptions<T>(size_a)) {
    const size_t n_vecs = utils::get_n_vecs_from_size(size);
    exponents = new uint8_t[n_vecs];
    factors = new uint8_t[n_vecs];
  }

  ~AlpCompressionData<T>() {
    delete[] exponents;
    delete[] factors;
  }
};

template <typename T> struct AlpRdCompressionData {
  using UINT_T = typename utils::same_width_uint<T>::type;

  size_t rowgroup_offset = 0;
  size_t size;

  AlpFFORArray<uint16_t> left_ffor;
  AlpFFORArray<UINT_T> right_ffor;

  uint16_t *left_parts_dicts;

  AlpExceptions<uint16_t> exceptions;

  AlpRdCompressionData<T>(const size_t size_a)
      : size(size_a), left_ffor(size_a), right_ffor(size_a),
        exceptions(AlpExceptions<uint16_t>(size_a)) {
    const size_t n_vecs = utils::get_n_vecs_from_size(size_a);
    left_parts_dicts = new uint16_t[n_vecs * config::MAX_RD_DICTIONARY_SIZE];
  }

  ~AlpRdCompressionData<T>() { delete[] left_parts_dicts; }
};

// Default ALP encoding
template <typename T>
void int_encode(const T *input_array, const size_t count,
                AlpCompressionData<T> *data);

// Default ALP decoding
template <typename T>
void int_decode(T *output_array, AlpCompressionData<T> *data);

// Rd ALP encoding
template <typename T>
void rd_encode(const T *input_array, const size_t count,
               AlpRdCompressionData<T> *data);

// Rd ALP decoding
template <typename T>
void rd_decode(T *output_array, AlpRdCompressionData<T> *data);

/*
// True ALP adaptive encoding
void encode();

// True ALP adaptive decoding
void decode();
*/

} // namespace alp

#endif // ALP_BINDINGS_HPP
