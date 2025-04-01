#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include "../alp/alp-bindings.cuh"
#include "../alp/constants.hpp"
#include "../fls/fls-bindings.hpp"

#ifndef DATA_CUH
#define DATA_CUH

namespace data {

namespace primitives {

template <typename T>
std::function<T()> get_random_number_generator(const T min, const T max) {
  std::random_device random_device;
  std::default_random_engine random_engine(random_device());
  std::uniform_int_distribution<T> uniform_dist(min, max);

  return std::bind(uniform_dist, random_engine);
}

template <typename T>
T *fill_array_with_constant(T *array, const size_t n_values, const T value) {
  for (size_t i{0}; i < n_values; ++i) {
    array[i] = value;
  }

  return array;
}

template <typename T>
T *fill_array_with_random_bytes(T *array, const size_t n_values,
                                const unsigned repeat = 1) {
  // WARNING Does not check if n_values % REPEAT == 0
  using UINT_T = typename utils::same_width_uint<T>::type;
  UINT_T *out = reinterpret_cast<UINT_T *>(array);
  auto generator = get_random_number_generator<UINT_T>(
      std::numeric_limits<UINT_T>::min(), std::numeric_limits<UINT_T>::max());

  for (size_t i{0}; i < n_values; i += repeat) {
    UINT_T value = generator();

    for (size_t r{0}; r < repeat; ++r) {
      out[i + r] = value;
    }
  }

  return array;
}

template <typename T>
T *fill_array_with_sequence(T *array, const size_t n_values, const T start,
                            const T step = 1, const unsigned repeat = 1) {
  // WARNING Does not check if n_values % REPEAT == 0
  using UINT_T = typename utils::same_width_uint<T>::type;
  UINT_T *out = reinterpret_cast<UINT_T *>(array);
  T value = start;

  for (size_t i{0}; i < n_values; i += repeat) {
    for (size_t r{0}; r < repeat; ++r) {
      out[i + r] = value;
    }

    value += step;
  }

  return array;
}

template <typename T>
T *fill_array_with_random_data(T *array, const size_t n_values,
                               const unsigned repeat = 1,
                               const T min = std::numeric_limits<T>::min(),
                               const T max = std::numeric_limits<T>::max()) {
  // WARNING Does not check if n_values % REPEAT == 0
  using UINT_T = typename utils::same_width_uint<T>::type;
  UINT_T *out = reinterpret_cast<UINT_T *>(array);
  auto generator = get_random_number_generator<UINT_T>(min, max);

  for (size_t i{0}; i < n_values; i += repeat) {
    UINT_T value = generator();

    for (size_t r{0}; r < repeat; ++r) {
      out[i + r] = value;
    }
  }

  return array;
}

template <typename T, typename T_sum>
T_sum sum_array(const T *array, const size_t n_values) {
  T_sum sum = 0;

  for (size_t i{0}; i < n_values; ++i) {
    sum += static_cast<T_sum>(array[i]);
  }

  return sum;
}

template <typename T_in, typename T_out>
T_out *prefix_sum_array(const T_in *in, T_out *out, const size_t n_values) {
  T_out sum = 0;

  for (size_t i{0}; i < n_values; ++i) {
    sum += in[i];
    out[i] = sum;
  }

  return out;
}

template <typename T>
std::tuple<T *, bool> make_column_magic(T *data, const size_t n_values) {
  auto generate_index = get_random_number_generator<size_t>(0, n_values - 1);
  auto generate_presence = get_random_number_generator<size_t>(0, 100);

  bool is_magic = generate_presence() < 50;
  if (is_magic) { // 50% chance of being magic
    data[generate_index()] = consts::as<T>::MAGIC_NUMBER;
  }

  return std::make_tuple(data, is_magic);
}

template <typename T>
T *generate_positions(T *positions, const T *counts, const size_t n_vecs) {
  // INFO Not really a primitive ...
  T *indices = new T[consts::VALUES_PER_VECTOR];
  for (size_t i{0}; i < consts::VALUES_PER_VECTOR; ++i) {
    indices[i] = i;
  }

  // Shuffle them and copy to the positions array
  std::random_device random_device;
  auto rng = std::default_random_engine(random_device());
  T *c_positions = positions;
  for (size_t vi{0}; vi < n_vecs; ++vi) {
    std::shuffle(std::begin(indices), std::end(indices), rng);
    std::memcpy(c_positions, indices, sizeof(T) * counts[vi]);
    std::sort(c_positions, c_positions + counts[vi]);
    c_positions += counts[vi];
  }
  delete[] indices;

  return positions;
}

} // namespace primitives

namespace generators {

template <typename T> struct ValueRange {
  const T min;
  const T max;

  ValueRange()
      : min(std::numeric_limits<T>::min()), max(std::numeric_limits<T>::max()) {
  }
  ValueRange(const T a_value) : min(a_value), max(a_value) {}
  ValueRange(const T a_start, const T a_end) : min(a_start), max(a_end) {}
};

namespace arrays {

template <typename T>
std::pair<T *, size_t> read_file_as(size_t input_count, std::string path) {

  // Open file
  std::ifstream inputFile(path, std::ios::binary | std::ios::ate);
  if (!inputFile) {
    throw std::invalid_argument("Could not open the specified file.");
  }
  // Get file size
  const std::streamsize file_size = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);

  // Check file size to contain right type of data
  bool file_size_is_multiple_of_T_size =
      static_cast<size_t>(file_size) % static_cast<size_t>(sizeof(T)) != 0;
  if (file_size_is_multiple_of_T_size) {
    throw std::invalid_argument(
        "File size is incorrect, it is not a multiple of the type's size.");
  }

  const size_t values_in_file = static_cast<size_t>(file_size) / sizeof(T);
  size_t count = input_count == 0 ? values_in_file : input_count;
  count = count - (count % consts::VALUES_PER_VECTOR);
  auto column = new T[count];

  // Read either the file size, or the total number of values needed,
  // whichever is smaller
  const std::streamsize read_size =
      std::min(file_size, static_cast<std::streamsize>((count * sizeof(T))));
  if (!inputFile.read(reinterpret_cast<char *>(column), read_size)) {
    throw std::invalid_argument("Failed to read file into column");
  }

  inputFile.close();

  // Copy paste the values in file until the column is filled
  if (values_in_file < count) {
    size_t n_filled_values = values_in_file;
    size_t n_empty_values_column = count - n_filled_values;
    while (n_empty_values_column > 0) {
      std::memcpy(column + n_filled_values, column,
                  std::min(n_empty_values_column, values_in_file));
      n_filled_values += values_in_file;

      if (n_empty_values_column < values_in_file) {
        break;
      }
      n_empty_values_column -= values_in_file;
    }
  }

  return std::make_pair(column, count);
}

template <typename T> T *generate_index_array(const size_t n_values, const vbw_t value_bit_width) {
	T mask = utils::h_set_first_n_bits<T>(value_bit_width);
  T *array =new T[n_values];

	for (size_t i{0}; i < n_values; i++) {
		array[i] = i & mask;
	}

  return array;
}

} // namespace arrays

namespace fls_bindings {

template <typename T>
flsgpu::host::BPColumn<T> compress(const T *array, const size_t n_values,
                                   const vbw_t value_bit_width) {
  size_t n_vecs = utils::get_n_vecs_from_size(n_values);
  size_t compressed_vector_size =
      utils::get_compressed_vector_size<T>(value_bit_width);
  size_t n_packed_values = n_vecs * compressed_vector_size;

  T *packed_array = new T[n_packed_values];
  T *c_packed_array = packed_array;

  for (size_t vi{0}; vi < n_vecs; ++vi) {
    fls::pack(array, c_packed_array, value_bit_width);
    array += consts::VALUES_PER_VECTOR;
    c_packed_array += compressed_vector_size;
  }

  return flsgpu::host::BPColumn<T>{
      n_values,
      n_packed_values,
      packed_array,
      primitives::fill_array_with_constant<vbw_t>(new vbw_t[n_vecs], n_vecs,
                                                  value_bit_width),
      primitives::fill_array_with_sequence<size_t>(
          new size_t[n_vecs], n_vecs, 0,
          utils::get_compressed_vector_size<T>(value_bit_width)),
  };
}

template <typename T> T *decompress(const flsgpu::host::BPColumn<T> column) {
  T *out_array = new T[column.n_values];
  T *c_out_array = out_array;
  T *c_packed_array = column.packed_array;

  for (size_t vi{0}; vi < column.get_n_vecs(); ++vi) {
    fls::unpack(c_packed_array, c_out_array, column.bit_widths[vi]);
    c_packed_array +=
        utils::get_compressed_vector_size<T>(column.bit_widths[vi]);
    c_out_array += consts::VALUES_PER_VECTOR;
  }

  return out_array;
}

template <typename T> T *decompress(const flsgpu::host::FFORColumn<T> column) {
  // WARNING Assumes vbw is constant for all vectors
  vbw_t value_bit_width = column.bit_widths[0];
  size_t n_vecs = utils::get_n_vecs_from_size(column.n_values);
  size_t compressed_vector_size =
      utils::get_compressed_vector_size<T>(value_bit_width);
  T *out_array = new T[column.n_values];
  T *c_out_array = out_array;
  T *c_packed_array = column.packed_array;

  for (size_t vi{0}; vi < n_vecs; ++vi) {
    fls::unffor(column.packed_array, c_out_array, value_bit_width,
                column.bases[vi]);
    c_packed_array += compressed_vector_size;
    c_out_array += consts::VALUES_PER_VECTOR;
  }

  return out_array;
}

} // namespace fls_bindings

namespace columns {

template <typename T>
flsgpu::host::BPColumn<T>
generate_index_bp_column(const size_t n_values, const vbw_t value_bit_width) {
  size_t n_vecs = utils::get_n_vecs_from_size(n_values);
  size_t n_packed_values =
      n_vecs * utils::get_compressed_vector_size<T>(value_bit_width);
  T *array = arrays::generate_index_array<T>(n_values);

  auto column = fls_bindings::compress<T>(array, n_values, value_bit_width);
  delete[] array;
  return column;
}

template <typename T>
flsgpu::host::BPColumn<T>
generate_random_bp_column(const size_t n_values,
                          const ValueRange<vbw_t> value_bit_width,
                          const int32_t repeat = 1) {
  size_t n_vecs = utils::get_n_vecs_from_size(n_values);
  vbw_t *bit_widths = primitives::fill_array_with_random_data<vbw_t>(
      new vbw_t[n_vecs], n_vecs, repeat, value_bit_width.min,
      value_bit_width.max);
  size_t n_packed_values =
      primitives::sum_array<vbw_t, size_t>(bit_widths, n_vecs) *
      utils::get_n_lanes<T>();

  return flsgpu::host::BPColumn<T>{
      n_values,
      n_packed_values,
      primitives::fill_array_with_random_bytes<T>(new T[n_packed_values],
                                                  n_packed_values),
      bit_widths,
      primitives::prefix_sum_array(bit_widths, new size_t[n_vecs], n_vecs),
  };
}

template <typename T>
flsgpu::host::FFORColumn<T> generate_random_ffor_column(
    const size_t n_values, const ValueRange<vbw_t> value_bit_width,
    const ValueRange<T> bases, const int32_t repeat = 1) {
  size_t n_vecs = utils::get_n_vecs_from_size(n_values);
  return flsgpu::host::FFORColumn<T>{
      generate_random_bp_column<T>(n_values, value_bit_width, repeat),
      primitives::fill_array_with_random_data<T>(new T[n_vecs], n_vecs, 1,
                                                 bases.min, bases.max),
  };
}

template <typename T>
flsgpu::host::ALPColumn<T> generate_alp_column(
    const size_t n_values, const ValueRange<uint16_t> exceptions_per_vec,
    const ValueRange<vbw_t> bit_width_range, const unsigned repeat = 1) {
  static_assert(std::is_floating_point<T>::value,
                "T should be a floating point type.");
  using UINT_T = typename utils::same_width_uint<T>::type;

  const size_t n_vecs = utils::get_n_vecs_from_size(n_values);
  auto column = new flsgpu::host::ALPColumn<T>();
  column.ffor = generate_random_ffor_column<UINT_T>(
      n_values, bit_width_range, repeat, ValueRange<UINT_T>(2, 20));

  // Note we halve the frac and fact because otherwise you
  // will have integer overflow in the decoding for some combinations
  int32_t frac_arr_size = sizeof(T) == 4 ? 11 : 21;
  int32_t fact_arr_size = sizeof(T) == 4 ? 10 : 19;
  column.fractions_indices = primitives::fill_array_with_random_data<uint8_t>(
      new uint8_t[n_vecs], n_vecs, repeat, 0,
      static_cast<uint8_t>(frac_arr_size / 2));
  column.factor_indices = primitives::fill_array_with_random_data<uint8_t>(
      new uint8_t[n_vecs], n_vecs, repeat, 0,
      static_cast<uint8_t>(fact_arr_size / 2));

  column.counts = primitives::fill_array_with_random_data<uint16_t>(
      new uint16_t[n_vecs], n_vecs, 1, exceptions_per_vec.min,
      exceptions_per_vec.max);

  column.n_exceptions =
      primitives::sum_array<uint16_t, size_t>(column.counts, n_vecs);
  column.exceptions_offsets =
      primitives::prefix_sum_array(column.counts, new size_t[n_vecs], n_vecs);
  column.exceptions = primitives::fill_array_with_random_bytes(
      new T[column.n_exceptions], n_values);
  column.positions = primitives::generate_positions<uint16_t>(
      new uint16_t[column.n_exceptions], column.counts, n_vecs);

  // Not supported (yet)
  column.compressed_size_bytes_alp = 0;
  column.compressed_size_bytes_alp_extended = 0;

  return column;
}

template <typename T>
void modify_alp_exception_count(flsgpu::host::ALPColumn<T> column,
                                const ValueRange<int32_t> exceptions_per_vec) {
  const size_t n_values = column.ffor.bp.n_values;
  const size_t n_vecs = utils::get_n_vecs_from_size(n_values);

  delete[] column.counts;
  delete[] column.exceptions_offsets;
  delete[] column.exceptions;
  delete[] column.positions;

  // Copied from generate_alp_column
  column.counts = primitives::fill_array_with_random_data<uint16_t>(
      new uint16_t[n_vecs], n_vecs, 1, exceptions_per_vec.min,
      exceptions_per_vec.max);
  column.n_exceptions =
      primitives::sum_array<uint16_t, size_t>(column.counts, n_vecs);
  column.exceptions_offsets =
      primitives::prefix_sum_array(column.counts, new size_t[n_vecs], n_vecs);
  column.exceptions = primitives::fill_array_with_random_bytes(
      new T[column.n_exceptions], n_values);
  column.positions = primitives::generate_positions<uint16_t>(
      new uint16_t[column.n_exceptions], column.counts, n_vecs);
}

template <typename T>
void modify_alp_value_bit_width(flsgpu::host::ALPColumn<T> column,
                                const ValueRange<vbw_t> bit_width_range,
                                const unsigned repeat = 1) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  flsgpu::host::free_column(column.ffor);
  column.ffor = generate_random_ffor_column<UINT_T>(column.ffor.bp.n_values,
                                                    bit_width_range, repeat,
                                                    ValueRange<UINT_T>(2, 20));
}

} // namespace columns

} // namespace generators
} // namespace data

#endif // DATA_CUH
