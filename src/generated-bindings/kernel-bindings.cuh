#include <cstdint>

#include "../engine/kernels.cuh"
#include "../flsgpu/flsgpu-api.cuh"

namespace generated_bindings {

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
T *decompress_column(ColumnT column, unsigned unpack_n_vectors,
                     unsigned unpack_n_values, Unpacker unpacker,
                     Patcher patcher);

template <typename T, typename ColumnT>
bool query_column(ColumnT column, unsigned unpack_n_vectors,
                  unsigned unpack_n_values, Unpacker unpacker, Patcher patcher);

template <typename T, typename ColumnT>
bool compute_column(ColumnT column, unsigned unpack_n_vectors,
                    unsigned unpack_n_values, Unpacker unpacker,
                    Patcher patcher);

template <typename T, typename ColumnT>
bool query_multi_column(ColumnT column, unsigned unpack_n_vectors,
                        unsigned unpack_n_values, Unpacker unpacker,
                        Patcher patcher, unsigned n_columns);

uint32_t *decompress_column(flsgpu::device::BPColumn<uint32_t> column,
                            unsigned unpack_n_vectors, unsigned unpack_n_values,
                            Unpacker unpacker, Patcher patcher);
uint64_t *decompress_column(flsgpu::device::BPColumn<uint64_t> column,
                            unsigned unpack_n_vectors, unsigned unpack_n_values,
                            Unpacker unpacker, Patcher patcher);
uint32_t *decompress_column(flsgpu::device::FFORColumn<uint32_t> column,
                            unsigned unpack_n_vectors, unsigned unpack_n_values,
                            Unpacker unpacker, Patcher patcher);
float *decompress_column(flsgpu::device::ALPColumn<float> column,
                         unsigned unpack_n_vectors, unsigned unpack_n_values,
                         Unpacker unpacker, Patcher patcher);
double *decompress_column(flsgpu::device::ALPColumn<double> column,
                          unsigned unpack_n_vectors, unsigned unpack_n_values,
                          Unpacker unpacker, Patcher patcher);
float *decompress_column(flsgpu::device::ALPExtendedColumn<float> column,
                         unsigned unpack_n_vectors, unsigned unpack_n_values,
                         Unpacker unpacker, Patcher patcher);
double *decompress_column(flsgpu::device::ALPExtendedColumn<double> column,
                          unsigned unpack_n_vectors, unsigned unpack_n_values,
                          Unpacker unpacker, Patcher patcher);

uint32_t *query_column(flsgpu::device::BPColumn<uint32_t> column,
                       unsigned unpack_n_vectors, unsigned unpack_n_values,
                       Unpacker unpacker, Patcher patcher);
uint64_t *query_column(flsgpu::device::BPColumn<uint64_t> column,
                       unsigned unpack_n_vectors, unsigned unpack_n_values,
                       Unpacker unpacker, Patcher patcher);
uint32_t *query_column(flsgpu::device::FFORColumn<uint32_t> column,
                       unsigned unpack_n_vectors, unsigned unpack_n_values,
                       Unpacker unpacker, Patcher patcher);
uint64_t *query_column(flsgpu::device::FFORColumn<uint64_t> column,
                       unsigned unpack_n_vectors, unsigned unpack_n_values,
                       Unpacker unpacker, Patcher patcher);
float *query_column(flsgpu::device::ALPColumn<float> column,
                    unsigned unpack_n_vectors, unsigned unpack_n_values,
                    Unpacker unpacker, Patcher patcher);
double *query_column(flsgpu::device::ALPColumn<double> column,
                     unsigned unpack_n_vectors, unsigned unpack_n_values,
                     Unpacker unpacker, Patcher patcher);
float *query_column(flsgpu::device::ALPExtendedColumn<float> column,
                    unsigned unpack_n_vectors, unsigned unpack_n_values,
                    Unpacker unpacker, Patcher patcher);
double *query_column(flsgpu::device::ALPExtendedColumn<double> column,
                     unsigned unpack_n_vectors, unsigned unpack_n_values,
                     Unpacker unpacker, Patcher patcher);

uint32_t *compute_column(flsgpu::device::FFORColumn<uint32_t> column,
                       unsigned unpack_n_vectors, unsigned unpack_n_values,
                       Unpacker unpacker, Patcher patcher);
uint64_t *compute_column(flsgpu::device::FFORColumn<uint64_t> column,
                       unsigned unpack_n_vectors, unsigned unpack_n_values,
                       Unpacker unpacker, Patcher patcher);
float *compute_column(flsgpu::device::ALPExtendedColumn<float> column,
                    unsigned unpack_n_vectors, unsigned unpack_n_values,
                    Unpacker unpacker, Patcher patcher);
double *compute_column(flsgpu::device::ALPExtendedColumn<double> column,
                     unsigned unpack_n_vectors, unsigned unpack_n_values,
                     Unpacker unpacker, Patcher patcher);

uint32_t *query_multi_column(flsgpu::device::FFORColumn<uint32_t> column,
                             unsigned unpack_n_vectors,
                             unsigned unpack_n_values, Unpacker unpacker,
                             Patcher patcher);
uint64_t *query_multi_column(flsgpu::device::FFORColumn<uint64_t> column,
                             unsigned unpack_n_vectors,
                             unsigned unpack_n_values, Unpacker unpacker,
                             Patcher patcher);
float *query_multi_column(flsgpu::device::ALPColumn<float> column,
                          unsigned unpack_n_vectors, unsigned unpack_n_values,
                          Unpacker unpacker, Patcher patcher);
double *query_multi_column(flsgpu::device::ALPColumn<double> column,
                           unsigned unpack_n_vectors, unsigned unpack_n_values,
                           Unpacker unpacker, Patcher patcher);
float *query_multi_column(flsgpu::device::ALPExtendedColumn<float> column,
                          unsigned unpack_n_vectors, unsigned unpack_n_values,
                          Unpacker unpacker, Patcher patcher);
double *query_multi_column(flsgpu::device::ALPExtendedColumn<double> column,
                           unsigned unpack_n_vectors, unsigned unpack_n_values,
                           Unpacker unpacker, Patcher patcher);

} // namespace generated_bindings
