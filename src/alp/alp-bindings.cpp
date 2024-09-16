#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "../common/utils.hpp"
#include "../fls/compression.hpp"
#include "alp-bindings.hpp"
#include "config.hpp"
#include "decoder.hpp"
#include "encoder.hpp"
#include "falp.hpp"
#include "rd.hpp"

namespace alp {
constexpr int MAX_ATTEMPTS_TO_ENCODE = 10000;

template <typename T>
void int_encode(const T *input_array, const size_t count,
                AlpCompressionData<T> *data) {
  using INT_T = typename utils::same_width_int<T>::type;
  using UINT_T = typename utils::same_width_uint<T>::type;

  const size_t n_vecs = utils::get_n_vecs_from_size(count);

  T *sample_array = new T[count];
  state<T> alpstate;

  int32_t attempts_to_int_encode = 0;
  while (attempts_to_int_encode < MAX_ATTEMPTS_TO_ENCODE) {
    alp::encoder<T>::init(input_array, data->rowgroup_offset, count,
                          sample_array, alpstate);
    if (alpstate.scheme == Scheme::ALP) {
      break;
    }
    ++attempts_to_int_encode;
  }
  if (attempts_to_int_encode >= 1000) {
    throw alp::EncodingException();
  }
  delete[] sample_array;

  INT_T *encoded_array = new INT_T[count];
  for (size_t i{0}; i < n_vecs; i++) {
    AlpVecExceptions<T> exceptions = data->exceptions.get_exceptions_for_vec(i);
    alp::encoder<T>::encode(input_array, exceptions.exceptions,
                            exceptions.positions, exceptions.count,
                            encoded_array, alpstate);
    data->exponents[i] = alpstate.exp;
    data->factors[i] = alpstate.fac;

    alp::encoder<T>::analyze_ffor(
        encoded_array, data->ffor.bit_widths[i],
        reinterpret_cast<INT_T *>(&data->ffor.bases[i]));

    fls::ffor(reinterpret_cast<UINT_T *>(encoded_array), data->ffor.array,
              data->ffor.bit_widths[i], &data->ffor.bases[i]);

    encoded_array += consts::VALUES_PER_VECTOR;
    data->ffor.array += consts::VALUES_PER_VECTOR;
    input_array += consts::VALUES_PER_VECTOR;
  }

  data->ffor.array -= consts::VALUES_PER_VECTOR * n_vecs;
  encoded_array -= consts::VALUES_PER_VECTOR * n_vecs;
  delete[] encoded_array;
}

template <typename T>
void int_decode(T *output_array, AlpCompressionData<T> *data) {
  const size_t n_vecs = utils::get_n_vecs_from_size(data->size);

  for (size_t i{0}; i < n_vecs; i++) {
    generated::falp::fallback::scalar::falp(
        data->ffor.array, output_array, data->ffor.bit_widths[i],
        &data->ffor.bases[i], data->factors[i], data->exponents[i]);

    AlpVecExceptions<T> exceptions = data->exceptions.get_exceptions_for_vec(i);
    alp::decoder<T>::patch_exceptions(output_array, exceptions.exceptions,
                                      exceptions.positions, exceptions.count);

    output_array += consts::VALUES_PER_VECTOR;
    data->ffor.array += consts::VALUES_PER_VECTOR;
  }

  data->ffor.array -= consts::VALUES_PER_VECTOR * n_vecs;
}

template <typename T>
void rd_encode(const T *input_array, const size_t count,
               AlpRdCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const size_t n_vecs = utils::get_n_vecs_from_size(count);
  auto left_parts_dicts = data->left_parts_dicts;

  T *sample_array = new T[count];
  state<T> alpstate;

  int32_t attempts_to_int_encode = 0;
  while (attempts_to_int_encode < MAX_ATTEMPTS_TO_ENCODE) {
    alp::encoder<T>::init(input_array, data->rowgroup_offset, count,
                          sample_array, alpstate);
    if (alpstate.scheme == Scheme::ALP_RD) {
      break;
    }
    ++attempts_to_int_encode;
  }
  if (attempts_to_int_encode >= 1000) {
    throw alp::EncodingException();
  }

  uint16_t left_array[consts::VALUES_PER_VECTOR];
  UINT_T right_array[consts::VALUES_PER_VECTOR];

  for (size_t i{0}; i < n_vecs; i++) {
    alp::rd_encoder<T>::init(const_cast<T *>(input_array),
                             data->rowgroup_offset, consts::VALUES_PER_VECTOR,
                             sample_array, alpstate);

    std::memcpy(left_parts_dicts, alpstate.left_parts_dict,
                config::MAX_RD_DICTIONARY_SIZE * sizeof(uint16_t));
    left_parts_dicts += config::MAX_RD_DICTIONARY_SIZE;

    AlpVecExceptions<uint16_t> exceptions =
        data->exceptions.get_exceptions_for_vec(i);
    AlpFFORVecHeader<uint16_t> left_ffor =
        data->left_ffor.get_ffor_header_for_vec(i);
    AlpFFORVecHeader<UINT_T> right_ffor =
        data->right_ffor.get_ffor_header_for_vec(i);

    alp::rd_encoder<T>::encode(input_array, exceptions.exceptions,
                               exceptions.positions, exceptions.count,
                               right_array, left_array, alpstate);

    (*left_ffor.base) = alpstate.left_for_base;
    (*left_ffor.bit_width) = alpstate.left_bit_width;
    (*right_ffor.base) = alpstate.right_for_base;
    (*right_ffor.bit_width) = alpstate.right_bit_width;

    fls::ffor(right_array, right_ffor.array, (*right_ffor.bit_width),
              right_ffor.base);
    fls::ffor(left_array, left_ffor.array, (*left_ffor.bit_width),
              left_ffor.base);

    input_array += consts::VALUES_PER_VECTOR;
  }

  delete[] sample_array;
}

template <typename T>
void rd_decode(T *output_array, AlpRdCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  const size_t n_vecs = utils::get_n_vecs_from_size(data->size);

  uint16_t left_array[consts::VALUES_PER_VECTOR];
  UINT_T right_array[consts::VALUES_PER_VECTOR];
  state<T> alpstate;

	uint16_t* left_parts_dict = data->left_parts_dicts;
  for (size_t i{0}; i < n_vecs; i++) {
    AlpFFORVecHeader<uint16_t> left_ffor_header =
        data->left_ffor.get_ffor_header_for_vec(i);
    AlpFFORVecHeader<UINT_T> right_ffor_header =
        data->right_ffor.get_ffor_header_for_vec(i);

    fls::unffor(left_ffor_header.array, left_array,
                (*left_ffor_header.bit_width), left_ffor_header.base);
    fls::unffor(right_ffor_header.array, right_array,
                (*right_ffor_header.bit_width), right_ffor_header.base);

    alpstate.right_bit_width = (*right_ffor_header.bit_width);

    for (size_t j{0}; j < config::MAX_RD_DICTIONARY_SIZE; j++) {
      alpstate.left_parts_dict[j] = left_parts_dict[j];
    }
    left_parts_dict += config::MAX_RD_DICTIONARY_SIZE;

    AlpVecExceptions<uint16_t> exceptions =
        data->exceptions.get_exceptions_for_vec(i);

    alp::rd_encoder<T>::decode(output_array, right_array, left_array,
                               exceptions.exceptions, exceptions.positions,
                               exceptions.count, alpstate);

    output_array += consts::VALUES_PER_VECTOR;
  }
}

} // namespace alp

template void alp::int_encode<float>(const float *input_array,
                                     const size_t count,
                                     alp::AlpCompressionData<float> *data);
template void alp::int_decode<float>(float *output_array,
                                     alp::AlpCompressionData<float> *data);

template void alp::int_encode<double>(const double *input_array,
                                      const size_t count,
                                      alp::AlpCompressionData<double> *data);
template void alp::int_decode<double>(double *output_array,
                                      alp::AlpCompressionData<double> *data);

template void alp::rd_encode(const float *input_array, const size_t count,
                             AlpRdCompressionData<float> *data);
template void alp::rd_decode(float *output_array,
                             AlpRdCompressionData<float> *data);

template void alp::rd_encode(const double *input_array, const size_t count,
                             AlpRdCompressionData<double> *data);
template void alp::rd_decode(double *output_array,
                             AlpRdCompressionData<double> *data);
