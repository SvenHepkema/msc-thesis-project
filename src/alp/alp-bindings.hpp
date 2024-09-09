#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../utils.hpp"
// WARNING The original ALP repo contains code that triggers warnings if all
// warnings are turned off. To make sure these warnings do not show up when the
// alp directory itself is not recompiled, I added this pragma to show it as a
// system header. So be carefult, warnings from the alp/* files do not show up
// when compiling
#pragma clang system_header

#ifndef ALP_BINDINGS_HPP
#define ALP_BINDINGS_HPP

namespace alp {

template<typename T>
struct AlpVecExceptions {
  T *exceptions;
  uint16_t *positions;
  uint16_t *count;
};

template<typename T>
struct AlpExceptions {
  T *exceptions;
  uint16_t *counts;
  uint16_t *positions;

	AlpExceptions<T>(const size_t size) {
		const size_t n_vecs = utils::get_n_vecs_from_size(size);
		// TODO: This should not allocate the maximum number of exceptions
		exceptions = new double[size];
		positions = new uint16_t[size];
		counts = new uint16_t[n_vecs];
	}

	AlpVecExceptions<T> get_exceptions_for_vec(const int64_t vec_index) {
		return AlpVecExceptions<T> {
			exceptions + vec_index * 1024,
			positions + vec_index * 1024,
			counts + vec_index,
		};
	}

	~AlpExceptions<T>() { 
		delete [] exceptions;
		delete [] positions;
		delete [] counts;
	}
};

template<typename T>
struct AlpCompressionData {
  using UINT_T = typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;
  using INT_T = typename std::conditional<sizeof(T) == 4, int32_t, int64_t>::type;
  size_t rowgroup_offset = 0;
  size_t size;

  UINT_T *ffor_array;      

  UINT_T * ffor_bases; 
  uint8_t* bit_widths;
  uint8_t* exponents;
  uint8_t* factors;

	AlpExceptions<T> exceptions;

	AlpCompressionData<T>(const size_t size_a) : size(size_a), exceptions(AlpExceptions<T>(size_a)) {
		// TODO: This should not allocate the maximum amount of space
		ffor_array = new uint64_t[size];

		const size_t n_vecs = utils::get_n_vecs_from_size(size);
		ffor_bases = new uint64_t[n_vecs];
		bit_widths = new uint8_t[n_vecs];
		exponents = new uint8_t[n_vecs];
		factors = new uint8_t[n_vecs];
	}

	~AlpCompressionData<T>() {
		delete[] ffor_array;

		delete[] ffor_bases;
		delete[] bit_widths;
		delete[] exponents;
		delete[] factors;
	}
};

// Default ALP encoding
template<typename T>
void int_encode(const T *input_array, const size_t count,
                AlpCompressionData<T> *data);

// Default ALP decoding
template<typename T>
void int_decode(T *output_array, AlpCompressionData<T> *data);

template<typename T>
void patch_exceptions(T *output_array, AlpCompressionData<T> *data);

/*
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
*/

} // namespace alp
	

#endif // ALP_BINDINGS_HPP
