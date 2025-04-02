#include <cstdint>

#include "../flsgpu/flsgpu-api.cuh"

#ifndef GENERATED_KERNEL_BINDINGS_CUH
#define GENERATED_KERNEL_BINDINGS_CUH

namespace bindings {

enum class Unpacker {
  Stateless,
  StatelessBranchless,
  StatefulCache,
  StatefulLocal1,
  StatefulLocal2,
  StatefulLocal4,
  StatefulRegister1,
  StatefulRegister2,
  StatefulRegister4,
  StatefulRegisterBranchless1,
  StatefulRegisterBranchless2,
  StatefulRegisterBranchless4,
  StatefulBranchless,
};

enum class Patcher {
  None,
  Stateless,
  Stateful,
  Naive,
  NaiveBranchless,
  PrefetchPosition,
  PrefetchAll,
  PrefetchAllBranchless,
};

template <typename T, typename ColumnT>
T *decompress_column(const ColumnT column, const unsigned unpack_n_vectors,
                     const unsigned unpack_n_values, const Unpacker unpacker,
                     const Patcher patcher);

template <typename T, typename ColumnT>
bool query_column(const ColumnT column, const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values, const Unpacker unpacker,
                  const Patcher patcher);

template <typename T, typename ColumnT>
bool compute_column(const ColumnT column, const unsigned unpack_n_vectors,
                    const unsigned unpack_n_values, const Unpacker unpacker,
                    const Patcher patcher);

template <typename T, typename ColumnT>
bool query_multi_column(ColumnT column, const unsigned unpack_n_vectors,
                        const unsigned unpack_n_values, const Unpacker unpacker,
                        const Patcher patcher, const unsigned n_columns);

uint32_t *decompress_column(const flsgpu::device::BPColumn<uint32_t> column,
                            const unsigned unpack_n_vectors,
                            const unsigned unpack_n_values,
                            const Unpacker unpacker, const Patcher patcher);
uint64_t *decompress_column(const flsgpu::device::BPColumn<uint64_t> column,
                            const unsigned unpack_n_vectors,
                            const unsigned unpack_n_values,
                            const Unpacker unpacker, const Patcher patcher);
uint32_t *decompress_column(const flsgpu::device::FFORColumn<uint32_t> column,
                            const unsigned unpack_n_vectors,
                            const unsigned unpack_n_values,
                            const Unpacker unpacker, const Patcher patcher);
float *decompress_column(const flsgpu::device::ALPColumn<float> column,
                         const unsigned unpack_n_vectors,
                         const unsigned unpack_n_values,
                         const Unpacker unpacker, const Patcher patcher);
double *decompress_column(const flsgpu::device::ALPColumn<double> column,
                          const unsigned unpack_n_vectors,
                          const unsigned unpack_n_values,
                          const Unpacker unpacker, const Patcher patcher);
float *decompress_column(const flsgpu::device::ALPExtendedColumn<float> column,
                         const unsigned unpack_n_vectors,
                         const unsigned unpack_n_values,
                         const Unpacker unpacker, const Patcher patcher);
bool decompress_column(const flsgpu::device::ALPExtendedColumn<double> column,
                       const unsigned unpack_n_vectors,
                       const unsigned unpack_n_values, const Unpacker unpacker,
                       const Patcher patcher);

bool query_column(const flsgpu::device::BPColumn<uint32_t> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values, const Unpacker unpacker,
                  const Patcher patcher);
bool query_column(const flsgpu::device::BPColumn<uint64_t> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values, const Unpacker unpacker,
                  const Patcher patcher);
bool query_column(const flsgpu::device::FFORColumn<uint32_t> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values, const Unpacker unpacker,
                  const Patcher patcher);
bool query_column(const flsgpu::device::FFORColumn<uint64_t> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values, const Unpacker unpacker,
                  const Patcher patcher);
bool query_column(const flsgpu::device::ALPColumn<float> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values, const Unpacker unpacker,
                  const Patcher patcher);
bool query_column(const flsgpu::device::ALPColumn<double> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values, const Unpacker unpacker,
                  const Patcher patcher);
bool query_column(const flsgpu::device::ALPExtendedColumn<float> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values, const Unpacker unpacker,
                  const Patcher patcher);
bool query_column(const flsgpu::device::ALPExtendedColumn<double> column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values, const Unpacker unpacker,
                  const Patcher patcher);

bool compute_column(const flsgpu::device::FFORColumn<uint32_t> column,
                    const unsigned unpack_n_vectors,
                    const unsigned unpack_n_values, const Unpacker unpacker,
                    const Patcher patcher);
bool compute_column(const flsgpu::device::FFORColumn<uint64_t> column,
                    const unsigned unpack_n_vectors,
                    const unsigned unpack_n_values, const Unpacker unpacker,
                    const Patcher patcher);

/*
bool query_multi_column(const flsgpu::device::FFORColumn<uint32_t> column,
                             const unsigned unpack_n_vectors,
                             const unsigned unpack_n_values,
                             const Unpacker unpacker, const Patcher patcher,
                             const unsigned n_columns);
bool query_multi_column(const flsgpu::device::FFORColumn<uint64_t> column,
                             const unsigned unpack_n_vectors,
                             const unsigned unpack_n_values,
                             const Unpacker unpacker, const Patcher patcher,
                             const unsigned n_columns);
bool query_multi_column(const flsgpu::device::ALPColumn<float> column,
                          const unsigned unpack_n_vectors,
                          const unsigned unpack_n_values,
                          const Unpacker unpacker, const Patcher patcher,
                          const unsigned n_columns);
bool query_multi_column(const flsgpu::device::ALPColumn<double> column,
                           const unsigned unpack_n_vectors,
                           const unsigned unpack_n_values,
                           const Unpacker unpacker, const Patcher patcher,
                           const unsigned n_columns);
bool query_multi_column(const flsgpu::device::ALPExtendedColumn<float> column,
                          const unsigned unpack_n_vectors,
                          const unsigned unpack_n_values,
                          const Unpacker unpacker, const Patcher patcher,
                          const unsigned n_columns);
bool
query_multi_column(const flsgpu::device::ALPExtendedColumn<double> column,
                   const unsigned unpack_n_vectors,
                   const unsigned unpack_n_values, const Unpacker unpacker,
                   const Patcher patcher, const unsigned n_columns);
                                                                         */

} // namespace bindings

#endif // GENERATED_KERNEL_BINDINGS_CUH
