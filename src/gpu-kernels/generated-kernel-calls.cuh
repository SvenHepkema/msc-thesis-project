
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>

#include "../alp/alp-bindings.hpp"
#include "../common/consts.hpp"
#include "../common/runspec.hpp"
#include "alp.cuh"
#include "host-alp-utils.cuh"
#include "host-utils.cuh"
#include "kernels-global.cuh"

#ifndef GENERATED_KERNEL_CALLS
#define GENERATED_KERNEL_CALLS

namespace generated_kernel_calls {

template <typename T>
void fls_decompress_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
    const T *in,
    const int32_t value_bit_width) {

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::decompress_column<                                   
        T, 1, 1, BitUnpackerStateless<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::decompress_column<                                   
        T, 4, 1, BitUnpackerStateless<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::decompress_column<                                   
        T, 1, 4, BitUnpackerStateless<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::decompress_column<                                   
        T, 4, 4, BitUnpackerStateless<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::decompress_column<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::decompress_column<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::decompress_column<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::decompress_column<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::decompress_column<                                   
        T, 1, 1, BitUnpackerStatelessBranchless<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::decompress_column<                                   
        T, 4, 1, BitUnpackerStatelessBranchless<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::decompress_column<                                   
        T, 1, 4, BitUnpackerStatelessBranchless<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::decompress_column<                                   
        T, 4, 4, BitUnpackerStatelessBranchless<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::decompress_column<                                   
        T, 1, 1, BitUnpackerStatefulBranchless<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::decompress_column<                                   
        T, 4, 1, BitUnpackerStatefulBranchless<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::decompress_column<                                   
        T, 1, 4, BitUnpackerStatefulBranchless<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::decompress_column<                                   
        T, 4, 4, BitUnpackerStatefulBranchless<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

}

template <typename T>
void fls_query_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
    const T *in,
    const int32_t value_bit_width
    ) {

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_column<                                   
        T, 1, 1, BitUnpackerStateless<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_column<                                   
        T, 4, 1, BitUnpackerStateless<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_column<                                   
        T, 1, 4, BitUnpackerStateless<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_column<                                   
        T, 4, 4, BitUnpackerStateless<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_column<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_column<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_column<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_column<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_column<                                   
        T, 1, 1, BitUnpackerStatelessBranchless<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_column<                                   
        T, 4, 1, BitUnpackerStatelessBranchless<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_column<                                   
        T, 1, 4, BitUnpackerStatelessBranchless<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_column<                                   
        T, 4, 4, BitUnpackerStatelessBranchless<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_column<                                   
        T, 1, 1, BitUnpackerStatefulBranchless<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_column<                                   
        T, 4, 1, BitUnpackerStatefulBranchless<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_column<                                   
        T, 1, 4, BitUnpackerStatefulBranchless<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_column<                                   
        T, 4, 4, BitUnpackerStatefulBranchless<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }

}

template <typename T>
void fls_compute_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
    const T *in,
    const int32_t value_bit_width) {

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::compute_column<                                   
        T, 1, 1, BitUnpackerStateless<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::compute_column<                                   
        T, 4, 1, BitUnpackerStateless<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::compute_column<                                   
        T, 1, 4, BitUnpackerStateless<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::compute_column<                                   
        T, 4, 4, BitUnpackerStateless<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::compute_column<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::compute_column<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::compute_column<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATEFUL == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::compute_column<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::compute_column<                                   
        T, 1, 1, BitUnpackerStatelessBranchless<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::compute_column<                                   
        T, 4, 1, BitUnpackerStatelessBranchless<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::compute_column<                                   
        T, 1, 4, BitUnpackerStatelessBranchless<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::compute_column<                                   
        T, 4, 4, BitUnpackerStatelessBranchless<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::compute_column<                                   
        T, 1, 1, BitUnpackerStatefulBranchless<T, 1, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::compute_column<                                   
        T, 4, 1, BitUnpackerStatefulBranchless<T, 4, 1, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::compute_column<                                   
        T, 1, 4, BitUnpackerStatefulBranchless<T, 1, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::compute_column<                                   
        T, 4, 4, BitUnpackerStatefulBranchless<T, 4, 4, BPFunctor<T>>>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }

}

template <typename T>
void alp_decompress_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
const alp::AlpCompressionData<T> *data
) {

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
            BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
            BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
            BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::decompress_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
            BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}

}

template <typename T>
void alp_query_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
const alp::AlpCompressionData<T> *data,
 const T magic_value
) {

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATELESS_WITH_SCANNER_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatelessWithScannerALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 1 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 1, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::STATEFUL_P == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            StatefulALPExceptionPatcher<T, 4, 4 >, 
            AlpColumn<T>>, AlpColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::NAIVE_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            NaiveBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_POSITION == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchPositionALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStateful<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStateful<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStateful<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatelessBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatelessBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatelessBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatelessBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 1, 
        AlpUnpacker<T, 1, 1,
        BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 1) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 1, 
        AlpUnpacker<T, 4, 1,
        BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 1 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 1 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 1, 4, 
        AlpUnpacker<T, 1, 4,
        BitUnpackerStatefulBranchless<T, 1, 4, ALPFunctor<T, 1>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 1, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && runspec::PREFETCH_ALL_BRANCHLESS == spec.patcher && spec.n_vecs == 4 && spec.n_vals == 4) {
    auto column = transfer::copy_alp_extended_column_to_gpu(data);
    kernels::device::alp::query_column<                                   
        T, 4, 4, 
        AlpUnpacker<T, 4, 4,
        BitUnpackerStatefulBranchless<T, 4, 4, ALPFunctor<T, 4>, consts::VALUES_PER_VECTOR>,
            PrefetchAllBranchlessALPExceptionPatcher<T, 4, 4 >, 
            AlpExtendedColumn<T>>, AlpExtendedColumn<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}

}

}
#endif // GENERATED_KERNEL_CALLS
