#include <cstdint>

#include "../engine/enums.cuh"
#include "../flsgpu/flsgpu-api.cuh"

#ifndef GENERATED_KERNEL_BINDINGS_CUH
#define GENERATED_KERNEL_BINDINGS_CUH

namespace bindings {

template <typename T, typename ColumnT>
T *decompress_column(const ColumnT column, const unsigned unpack_n_vectors,
                     const unsigned unpack_n_values,
                     const enums::Unpacker unpacker,
                     const enums::Patcher patcher);

template <typename T, typename ColumnT>
bool query_column(const ColumnT column, const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values,
                  const enums::Unpacker unpacker, const enums::Patcher patcher,
                  const T magic_value);

template <typename T, typename ColumnT>
bool compute_column(const ColumnT column, const unsigned unpack_n_vectors,
                    const unsigned unpack_n_values,
                    const enums::Unpacker unpacker,
                    const enums::Patcher patcher, const unsigned n_repetitions);

template <typename T, typename ColumnT>
bool query_multi_column(ColumnT column, const unsigned unpack_n_vectors,
                        const unsigned unpack_n_values,
                        const enums::Unpacker unpacker,
                        const enums::Patcher patcher, const unsigned n_columns);

uint32_t *decompress_column(const flsgpu::device::BPColumn<uint32_t> column,
                            const unsigned unpack_n_vectors,
                            const unsigned unpack_n_values,
                            const enums::Unpacker unpacker,
                            const enums::Patcher patcher);
uint64_t *decompress_column(const flsgpu::device::BPColumn<uint64_t> column,
                            const unsigned unpack_n_vectors,
                            const unsigned unpack_n_values,
                            const enums::Unpacker unpacker,
                            const enums::Patcher patcher);
uint32_t *decompress_column(const flsgpu::device::FFORColumn<uint32_t> column,
                            const unsigned unpack_n_vectors,
                            const unsigned unpack_n_values,
                            const enums::Unpacker unpacker,
                            const enums::Patcher patcher);
uint64_t *decompress_column(const flsgpu::device::FFORColumn<uint64_t> column,
                            const unsigned unpack_n_vectors,
                            const unsigned unpack_n_values,
                            const enums::Unpacker unpacker,
                            const enums::Patcher patcher);
float *decompress_column(const flsgpu::device::ALPColumn<float> column,
                         const unsigned unpack_n_vectors,
                         const unsigned unpack_n_values,
                         const enums::Unpacker unpacker,
                         const enums::Patcher patcher);
double *decompress_column(const flsgpu::device::ALPColumn<double> column,
                          const unsigned unpack_n_vectors,
                          const unsigned unpack_n_values,
                          const enums::Unpacker unpacker,
                          const enums::Patcher patcher);
float *decompress_column(const flsgpu::device::ALPExtendedColumn<float> column,
                         const unsigned unpack_n_vectors,
                         const unsigned unpack_n_values,
                         const enums::Unpacker unpacker,
                         const enums::Patcher patcher);
double *
decompress_column(const flsgpu::device::ALPExtendedColumn<double> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values,
                  const enums::Unpacker unpacker, const enums::Patcher patcher);

bool query_column(const flsgpu::device::BPColumn<uint32_t> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values,
                  const enums::Unpacker unpacker, const enums::Patcher patcher,
                  const uint32_t magic_value);
bool query_column(const flsgpu::device::BPColumn<uint64_t> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values,
                  const enums::Unpacker unpacker, const enums::Patcher patcher,
                  const uint64_t magic_value);
bool query_column(const flsgpu::device::FFORColumn<uint32_t> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values,
                  const enums::Unpacker unpacker, const enums::Patcher patcher,
                  const uint32_t magic_value);
bool query_column(const flsgpu::device::FFORColumn<uint64_t> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values,
                  const enums::Unpacker unpacker, const enums::Patcher patcher,
                  const uint64_t magic_value);
bool query_column(const flsgpu::device::ALPColumn<float> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values,
                  const enums::Unpacker unpacker, const enums::Patcher patcher,
                  const float magic_value);
bool query_column(const flsgpu::device::ALPColumn<double> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values,
                  const enums::Unpacker unpacker, const enums::Patcher patcher,
                  const double magic_value);
bool query_column(const flsgpu::device::ALPExtendedColumn<float> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values,
                  const enums::Unpacker unpacker, const enums::Patcher patcher,
                  const float magic_value);
bool query_column(const flsgpu::device::ALPExtendedColumn<double> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values,
                  const enums::Unpacker unpacker, const enums::Patcher patcher,
                  const double magic_value);

bool compute_column(const flsgpu::device::FFORColumn<uint32_t> column,
                    const unsigned unpack_n_vectors,
                    const unsigned unpack_n_values,
                    const enums::Unpacker unpacker,
                    const enums::Patcher patcher, const unsigned n_repetitions);
bool compute_column(const flsgpu::device::FFORColumn<uint64_t> column,
                    const unsigned unpack_n_vectors,
                    const unsigned unpack_n_values,
                    const enums::Unpacker unpacker,
                    const enums::Patcher patcher, const unsigned n_repetitions);

/*
bool query_multi_column(const flsgpu::device::FFORColumn<uint32_t> column,
                             const unsigned unpack_n_vectors,
                             const unsigned unpack_n_values,
                             const enums::Unpacker unpacker, const
enums::Patcher patcher, const unsigned n_columns); bool query_multi_column(const
flsgpu::device::FFORColumn<uint64_t> column, const unsigned unpack_n_vectors,
                             const unsigned unpack_n_values,
                             const enums::Unpacker unpacker, const
enums::Patcher patcher, const unsigned n_columns); bool query_multi_column(const
flsgpu::device::ALPColumn<float> column, const unsigned unpack_n_vectors, const
unsigned unpack_n_values, const enums::Unpacker unpacker, const enums::Patcher
patcher, const unsigned n_columns); bool query_multi_column(const
flsgpu::device::ALPColumn<double> column, const unsigned unpack_n_vectors, const
unsigned unpack_n_values, const enums::Unpacker unpacker, const enums::Patcher
patcher, const unsigned n_columns); bool query_multi_column(const
flsgpu::device::ALPExtendedColumn<float> column, const unsigned
unpack_n_vectors, const unsigned unpack_n_values, const enums::Unpacker
unpacker, const enums::Patcher patcher, const unsigned n_columns); bool
query_multi_column(const flsgpu::device::ALPExtendedColumn<double> column,
                   const unsigned unpack_n_vectors,
                   const unsigned unpack_n_values, const enums::Unpacker
unpacker, const enums::Patcher patcher, const unsigned n_columns);
                                                                         */

} // namespace bindings

#endif // GENERATED_KERNEL_BINDINGS_CUH
