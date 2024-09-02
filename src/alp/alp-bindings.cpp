#include <stdexcept>

#include "alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "decode.hpp"
#include "encode.hpp"
#include "falp.hpp"
#include "rd.hpp"

namespace alp {

void int_encode(const double *input_array, const size_t count,
                AlpCompressionData *data) {
  alp::AlpEncode<double>::init(input_array, data->rowgroup_offset, count,
                               data->sample_array, data->state);
  alp::AlpEncode<double>::encode(
      input_array, data->exceptions_array, data->exceptions_position_array,
      data->exceptions_count_array, data->encoded_array, data->state);
  alp::AlpEncode<double>::analyze_ffor(data->encoded_array, data->bit_width,
                                       data->ffor_base_array);
  fls::ffor(data->encoded_array, data->ffor_array, data->bit_width,
             data->ffor_base_array);
}

void int_decode(double *output_array, AlpCompressionData *data) {
  generated::falp::fallback::scalar::falp(
      reinterpret_cast<uint64_t *>(data->ffor_array), output_array,
      data->bit_width, reinterpret_cast<uint64_t *>(data->ffor_base_array),
      data->state.fac, data->state.exp);
  alp::AlpDecode<double>::patch_exceptions(output_array, data->exceptions_array,
                                           data->exceptions_position_array,
                                           data->exceptions_count_array);
}

void rd_encode() { throw std::logic_error("Not implemented yet"); }

void rd_decode() { throw std::logic_error("Not implemented yet"); }

void encode() { throw std::logic_error("Not implemented yet"); }

void decode() { throw std::logic_error("Not implemented yet"); }

} // namespace alp
