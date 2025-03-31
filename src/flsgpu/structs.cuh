#include <cstddef>
#include <cstdint>

#include "device-types.cuh"
#include "host-utils.cuh"
#include "utils.cuh"

#ifndef STRUCTS_CUH 
#define STRUCTS_CUH 

namespace flsgpu {

namespace device {

template <typename T> struct BPColumn {
  using UINT_T = typename utils::same_width_uint<T>::type;
  size_t n_values;
  size_t n_vecs;

  UINT_T *packed_array;
  vbw_t *bit_widths;
  size_t *vector_offsets;
};

template <typename T> struct FFORColumn {
  using UINT_T = typename utils::same_width_uint<T>::type;
  BPColumn<T> bp;
  UINT_T *bases;
};

template <typename T> struct ALPColumn {
  FFORColumn<T> ffor;

  uint8_t *factor_indices;
  uint8_t *fraction_indices;

  size_t n_exceptions;
  T *exceptions;
  uint16_t *positions;
  uint16_t *counts;
};

template <typename T> struct ALPExtendedColumn {
  FFORColumn<T> ffor;

  uint8_t *factor_indices;
  uint8_t *fraction_indices;

  size_t n_exceptions;
  T *exceptions;
  uint16_t *positions;
  uint16_t *offsets_counts;
};

} // namespace device
namespace host {

template <typename T> struct BPColumn {
  using UINT_T = typename utils::same_width_uint<T>::type;
  size_t n_values;
  size_t n_packed_values;
  size_t n_vecs() const { return utils::get_n_vecs_from_size(n_values); }

  UINT_T *packed_array;
  vbw_t *bit_widths;
  size_t *vector_offsets;

  device::BPColumn<T> copy_to_device() {
    return device::BPColumn<T>{
        n_values, n_vecs(),
        GPUArray<UINT_T>(n_packed_values, packed_array).release(),
        GPUArray<vbw_t>(n_vecs(), bit_widths).release(),
        GPUArray<size_t>(n_vecs(), vector_offsets).release()};
  }
};

template <typename T> struct FFORColumn {
  using UINT_T = typename utils::same_width_uint<T>::type;

  BPColumn<T> bp;
  UINT_T *bases;

  device::FFORColumn<T> copy_to_device() {
    return device::FFORColumn<T>{
        bp.copy_to_device(), GPUArray<UINT_T>(bp.n_vecs(), bases).release()};
  }
};

template <typename T> struct ALPColumn {
  FFORColumn<T> ffor;

  uint8_t *factor_indices;
  uint8_t *fraction_indices;

  size_t n_exceptions;
  T *exceptions;
  uint16_t *positions;
  uint16_t *counts;

  device::ALPColumn<T> copy_to_device() {
    return device::ALPColumn<T>{
        ffor.copy_to_device(),
        GPUArray<uint8_t>(ffor.bp.n_vecs(), factor_indices).release(),
        GPUArray<uint8_t>(ffor.bp.n_vecs(), fraction_indices).release(),
        n_exceptions,
        GPUArray<T>(n_exceptions, exceptions).release(),
        GPUArray<uint16_t>(n_exceptions, positions).release(),
        GPUArray<uint16_t>(ffor.bp.n_vecs(), counts).release(),
    };
  }
};

template <typename T> struct ALPExtendedColumn {
  FFORColumn<T> ffor;

  uint8_t *factor_indices;
  uint8_t *fraction_indices;

  size_t n_exceptions;
  T *exceptions;
  uint16_t *positions;
  uint16_t *offsets_counts;

  device::ALPExtendedColumn<T> copy_to_device() {
    return device::ALPExtendedColumn<T>{
        ffor.copy_to_device(),
        GPUArray<uint8_t>(ffor.bp.n_vecs(), factor_indices).release(),
        GPUArray<uint8_t>(ffor.bp.n_vecs(), fraction_indices).release(),
        n_exceptions,
        GPUArray<T>(n_exceptions, exceptions).release(),
        GPUArray<uint16_t>(n_exceptions, positions).release(),
        GPUArray<uint16_t>(ffor.bp.n_vecs() * utils::get_n_lanes<T>(),
                           offsets_counts)
            .release(),
    };
  }
};

// Structs that are passed to the GPU cannot contain methods,
// that is why there is a separate method for the destructors
template <typename T> void destroy_column(device::BPColumn<T> column) {
  free_device_pointer(column.packed_array);
  free_device_pointer(column.bit_widths);
  free_device_pointer(column.vector_offsets);
}

template <typename T> void destroy_column(device::FFORColumn<T> column) {
  destroy_column(column.bp);
  free_device_pointer(column.bases);
}

template <typename T> void destroy_column(device::ALPColumn<T> column) {
  destroy_column(column.ffor);
  free_device_pointer(column.factor_indices);
  free_device_pointer(column.fraction_indices);
  free_device_pointer(column.exceptions);
  free_device_pointer(column.positions);
  free_device_pointer(column.counts);
}

template <typename T> void destroy_column(device::ALPExtendedColumn<T> column) {
  destroy_column(column.ffor);
  free_device_pointer(column.factor_indices);
  free_device_pointer(column.fraction_indices);
  free_device_pointer(column.exceptions);
  free_device_pointer(column.positions);
  free_device_pointer(column.offsets_counts);
}

} // namespace host
} // namespace flsgpu

#endif // STRUCTS_CUH 
