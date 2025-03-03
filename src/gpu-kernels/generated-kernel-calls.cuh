
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

}

template <typename T>
void fls_query_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
    const T *in,
    const int32_t value_bit_width
    ) {

}

template <typename T>
void fls_query_multicolumn(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
    const T *in_a,
    const T *in_b,
    const T *in_c,
    const T *in_d,
    const int32_t value_bit_width
    ) {

if (runspec::DUMMY == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, kernels::device::fls::Dummy<T, 1, 1, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::DUMMY == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, kernels::device::fls::Dummy<T, 4, 1, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::DUMMY == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, kernels::device::fls::Dummy<T, 1, 4, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::DUMMY == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, kernels::device::fls::Dummy<T, 4, 4, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::DUMMY == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 32) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 32, kernels::device::fls::Dummy<T, 1, 32, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::DUMMY == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 32) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 32, kernels::device::fls::Dummy<T, 4, 32, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::OLD_FLS_ADJUSTED == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 32) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 32, kernels::device::fls::OldFLSAdjusted<T, 1, 32, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::NON_INTERLEAVED == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerNonInterleaved<T, 1, 1, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStateless<T, 1, 1, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_CACHE == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T> , CacheLoader<T, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_1 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T> , LocalMemoryLoader<T, 1, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_2 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T> , LocalMemoryLoader<T, 1, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_4 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T> , LocalMemoryLoader<T, 1, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_1 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T> , RegisterLoader<T, 1, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_2 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T> , RegisterLoader<T, 1, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_4 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T> , RegisterLoader<T, 1, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_1 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T> , RegisterBranchlessLoader<T, 1, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_2 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T> , RegisterBranchlessLoader<T, 1, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_4 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStateful<T, 1, 1, BPFunctor<T> , RegisterBranchlessLoader<T, 1, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStatelessBranchless<T, 1, 1, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 1, BitUnpackerStatefulBranchless<T, 1, 1, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::NON_INTERLEAVED == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerNonInterleaved<T, 4, 1, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStateless<T, 4, 1, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_CACHE == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T> , CacheLoader<T, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_1 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T> , LocalMemoryLoader<T, 4, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_2 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T> , LocalMemoryLoader<T, 4, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_4 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T> , LocalMemoryLoader<T, 4, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_1 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T> , RegisterLoader<T, 4, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_2 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T> , RegisterLoader<T, 4, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_4 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T> , RegisterLoader<T, 4, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_1 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T> , RegisterBranchlessLoader<T, 4, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_2 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T> , RegisterBranchlessLoader<T, 4, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_4 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStateful<T, 4, 1, BPFunctor<T> , RegisterBranchlessLoader<T, 4, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStatelessBranchless<T, 4, 1, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 1) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 1, BitUnpackerStatefulBranchless<T, 4, 1, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::NON_INTERLEAVED == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerNonInterleaved<T, 1, 4, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStateless<T, 1, 4, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_CACHE == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T> , CacheLoader<T, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_1 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T> , LocalMemoryLoader<T, 1, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_2 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T> , LocalMemoryLoader<T, 1, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_4 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T> , LocalMemoryLoader<T, 1, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_1 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T> , RegisterLoader<T, 1, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_2 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T> , RegisterLoader<T, 1, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_4 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T> , RegisterLoader<T, 1, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_1 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T> , RegisterBranchlessLoader<T, 1, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_2 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T> , RegisterBranchlessLoader<T, 1, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_4 == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStateful<T, 1, 4, BPFunctor<T> , RegisterBranchlessLoader<T, 1, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStatelessBranchless<T, 1, 4, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 1 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 1, 4, BitUnpackerStatefulBranchless<T, 1, 4, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::NON_INTERLEAVED == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerNonInterleaved<T, 4, 4, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATELESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStateless<T, 4, 4, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_CACHE == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T> , CacheLoader<T, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_1 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T> , LocalMemoryLoader<T, 4, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_2 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T> , LocalMemoryLoader<T, 4, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_LOCAL_MEMORY_4 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T> , LocalMemoryLoader<T, 4, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_1 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T> , RegisterLoader<T, 4, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_2 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T> , RegisterLoader<T, 4, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_4 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T> , RegisterLoader<T, 4, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_1 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T> , RegisterBranchlessLoader<T, 4, 1> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_2 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T> , RegisterBranchlessLoader<T, 4, 2> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_REGISTER_BRANCHLESS_4 == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStateful<T, 4, 4, BPFunctor<T> , RegisterBranchlessLoader<T, 4, 4> >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATELESS_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStatelessBranchless<T, 4, 4, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

if (runspec::STATEFUL_BRANCHLESS == spec.unpacker && spec.n_vecs == 4 && spec.n_vals == 4) {
    kernels::device::fls::query_multicolumn<                                   
        T, 4, 4, BitUnpackerStatefulBranchless<T, 4, 4, BPFunctor<T>  >>           
        <<<n_blocks, n_threads>>>(                 
            out, in_a, in_b, in_c, in_d, 
            value_bit_width, value_bit_width, value_bit_width, value_bit_width);            
        }

}

template <typename T>
void fls_query_column_unrolled(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
    const T *in,
    const int32_t value_bit_width
    ) {

}

template <typename T>
void fls_compute_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
    const T *in,
    const int32_t value_bit_width) {

}

template <typename T>
void alp_decompress_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
const alp::AlpCompressionData<T> *data
) {

}

template <typename T>
void alp_query_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
const alp::AlpCompressionData<T> *data,
 const T magic_value
) {

}

}
#endif // GENERATED_KERNEL_CALLS
