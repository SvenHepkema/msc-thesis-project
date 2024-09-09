#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "../fls/compression.hpp"
#include "../utils.hpp"
#include "alp-bindings.hpp"
#include "decode.hpp"
#include "encode.hpp"
#include "falp.hpp"
#include "rd.hpp"
#include "state.hpp"

namespace alp {

template <typename T>
void int_encode(const T *input_array, const size_t count,
                AlpCompressionData<T> *data) {
  using INT_T =
      typename std::conditional<sizeof(T) == 4, int32_t, int64_t>::type;
  using UINT_T =
      typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;

  const size_t n_vecs = utils::get_n_vecs_from_size(count);

  T *sample_array = new double[count];
  state alpstate;

  int32_t attempts_to_int_encode = 0;
  while (attempts_to_int_encode < 1000) {
    alp::AlpEncode<T>::init(input_array, data->rowgroup_offset, count,
                            sample_array, alpstate);
    if (alpstate.scheme == SCHEME::ALP) {
      break;
		}
    ++attempts_to_int_encode;
  }
	if (attempts_to_int_encode >= 1000) {
		throw std::logic_error("Could not encode data as alp int\n");
	}
  delete[] sample_array;

  INT_T *encoded_array = new INT_T[count];

  for (size_t i{0}; i < n_vecs; i++) {
    alp::AlpEncode<T>::encode(input_array, data->exceptions.exceptions,
                              data->exceptions.positions,
                              data->exceptions.counts, encoded_array, alpstate);
    data->exponents[i] = alpstate.exp;
    data->factors[i] = alpstate.fac;

    alp::AlpEncode<T>::analyze_ffor(
        encoded_array, data->bit_widths[i],
        reinterpret_cast<INT_T *>(&data->ffor_bases[i]));

		/*
printf("%d, %d, %d, %d, %f\n", data->bit_widths[i],
data->exceptions.counts[0], alpstate.exp, alpstate.fac,
input_array[0]);
*/

    fls::ffor(reinterpret_cast<UINT_T *>(encoded_array), data->ffor_array,
              data->bit_widths[i], &data->ffor_bases[i]);

    encoded_array += consts::VALUES_PER_VECTOR;
    data->exceptions.add_offset(consts::VALUES_PER_VECTOR);
    data->ffor_array += consts::VALUES_PER_VECTOR;
    input_array += consts::VALUES_PER_VECTOR;
  }

  data->exceptions.add_offset(
      -static_cast<int64_t>(consts::VALUES_PER_VECTOR * n_vecs));
  data->ffor_array -= consts::VALUES_PER_VECTOR * n_vecs;
  encoded_array -= consts::VALUES_PER_VECTOR * n_vecs;
  delete[] encoded_array;
}

template <typename T>
void int_decode(T *output_array, AlpCompressionData<T> *data) {
  const size_t n_vecs = utils::get_n_vecs_from_size(data->size);

  for (size_t i{0}; i < n_vecs; i++) {
    generated::falp::fallback::scalar::falp(
        data->ffor_array, output_array, data->bit_widths[i],
        &data->ffor_bases[i], data->factors[i], data->exponents[i]);


    alp::AlpDecode<T>::patch_exceptions(
        output_array, data->exceptions.exceptions, data->exceptions.positions,
        data->exceptions.counts);

    output_array += consts::VALUES_PER_VECTOR;
    data->ffor_array += consts::VALUES_PER_VECTOR;
    data->exceptions.add_offset(consts::VALUES_PER_VECTOR);
  }

  data->exceptions.add_offset(
      -static_cast<int64_t>(consts::VALUES_PER_VECTOR * n_vecs));
  data->ffor_array -= consts::VALUES_PER_VECTOR * n_vecs;
}

template<typename T>
void patch_exceptions(T *output_array, AlpCompressionData<T> *data) {
  const size_t n_vecs = utils::get_n_vecs_from_size(data->size);

  for (size_t i{0}; i < n_vecs; i++) {
    alp::AlpDecode<T>::patch_exceptions(
        output_array, data->exceptions.exceptions, data->exceptions.positions,
        data->exceptions.counts);

    output_array += consts::VALUES_PER_VECTOR;
    data->ffor_array += consts::VALUES_PER_VECTOR;
    data->exceptions.add_offset(consts::VALUES_PER_VECTOR);
  }

  data->exceptions.add_offset(
      -static_cast<int64_t>(consts::VALUES_PER_VECTOR * n_vecs));
  data->ffor_array -= consts::VALUES_PER_VECTOR * n_vecs;
}

} // namespace alp

// template
// void alp::int_encode(float *output_array, alp::AlpCompressionData<float>
// *data); template void alp::int_decode(float *output_array,
// alp::AlpCompressionData<float> *data);
template void alp::int_encode<double>(const double *input_array,
                                      const size_t count,
                                      alp::AlpCompressionData<double> *data);
template void alp::int_decode<double>(double *output_array,
                                      alp::AlpCompressionData<double> *data);
template void alp::patch_exceptions<double>(double *output_array,
                                      alp::AlpCompressionData<double> *data);
