#include <cstddef>
#include <cstdint>

#include "state.hpp"

#ifndef ALP_BINDINGS_HPP
#define ALP_BINDINGS_HPP

namespace alp {

struct AlpCompressionData {
  size_t rowgroup_offset = 0;

  double *sample_array;
  int64_t *encoded_array;   // why int and not uint?
  int64_t *ffor_base_array; // why int and not uint?
  int64_t *ffor_array;      // why int and not uint?
  uint8_t bit_width;

  double *exceptions_array;
  uint16_t *exceptions_count_array;
  uint16_t *exceptions_position_array;

  state state;
};

// Default ALP encoding
void int_encode(const double *input_array, const size_t count,
                AlpCompressionData *data);

// Default ALP decoding
void int_decode(double *output_array, AlpCompressionData *data);

// falp step
// void falp();

// patching step
// void patch_exceptions();

// Rd ALP encoding
void rd_encode();

// Rd ALP decoding
void rd_decode();

// True ALP adaptive encoding
void encode();

// True ALP adaptive decoding
void decode();

} // namespace alp
#endif // ALP_BINDINGS_HPP
