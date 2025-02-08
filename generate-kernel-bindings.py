#!/usr/bin/python3

import os
import sys

import argparse
import logging
from typing import Any, NewType
import math

from dataclasses import dataclass

Code = NewType("Code", str)


@dataclass
class Parameter:
    tag: str
    values: list[Any]


@dataclass
class MultiParameter:
    parameters: list[Parameter]

    def __post_init__(self):
        equal_length = len(self.parameters[0].values)
        for parameter in self.parameters:
            assert equal_length == len(parameter.values)


FILE_HEADER = """
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
"""

FILE_FOOTER = """
}
#endif // GENERATED_KERNEL_CALLS
"""

FUNCTION_FOOTER = """
}
"""

FLS_DECOMPRESS_COLUMN_FUNCTION_SIGNATURE = """
template <typename T>
void fls_decompress_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
    const T *in,
    const int32_t value_bit_width) {
"""

FLS_DECOMPRESS_COLUMN_IF_STATEMENT = """
if (runspec::XXUNPACKER_ENUM == spec.unpacker && spec.n_vecs == XXN_VEC && spec.n_vals == XXN_VAL) {
    kernels::device::fls::decompress_column<                                   
        T, XXN_VEC, XXN_VAL, XXUNPACKER_T>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }
"""

FLS_QUERY_COLUMN_FUNCTION_SIGNATURE = """
template <typename T>
void fls_query_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
    const T *in,
    const int32_t value_bit_width
    ) {
"""

FLS_QUERY_COLUMN_IF_STATEMENT = """
if (runspec::XXUNPACKER_ENUM == spec.unpacker && spec.n_vecs == XXN_VEC && spec.n_vals == XXN_VAL) {
    kernels::device::fls::query_column<                                   
        T, XXN_VEC, XXN_VAL, XXUNPACKER_T>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width);            
        }
"""

FLS_COMPUTE_COLUMN_FUNCTION_SIGNATURE = """
template <typename T>
void fls_compute_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
    const T *in,
    const int32_t value_bit_width) {
"""

FLS_COMPUTE_COLUMN_IF_STATEMENT = """
if (runspec::XXUNPACKER_ENUM == spec.unpacker && spec.n_vecs == XXN_VEC && spec.n_vals == XXN_VAL) {
    kernels::device::fls::compute_column<                                   
        T, XXN_VEC, XXN_VAL, XXUNPACKER_T>           
        <<<n_blocks, n_threads>>>(                 
            out, in, value_bit_width, 0);            
        }
"""

ALP_DECOMPRESS_COLUMN_FUNCTION_SIGNATURE = """
template <typename T>
void alp_decompress_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
const alp::AlpCompressionData<T> *data
) {
"""

ALP_DECOMPRESS_COLUMN_IF_STATEMENT = """
if (runspec::XXUNPACKER_ENUM == spec.unpacker && runspec::XXPATCHER_ENUM == spec.patcher && spec.n_vecs == XXN_VEC && spec.n_vals == XXN_VAL) {
    XXCOLUMN_COPY
    kernels::device::alp::decompress_column<                                   
        T, XXN_VEC, XXN_VAL, 
        AlpUnpacker<T, XXN_VEC, XXN_VAL,
            XXUNPACKER_T,
            XXPATCHER_T<T, XXN_VEC, XXN_VAL >, 
            XXCOLUMN_T<T>>, XXCOLUMN_T<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column);            
    transfer::destroy_alp_column(column);
}
"""

ALP_QUERY_COLUMN_FUNCTION_SIGNATURE = """
template <typename T>
void alp_query_column(const runspec::KernelSpecification spec,
    const unsigned n_blocks, const unsigned n_threads,
    T *out,
const alp::AlpCompressionData<T> *data,
 const T magic_value
) {
"""
ALP_QUERY_COLUMN_IF_STATEMENT = """
if (runspec::XXUNPACKER_ENUM == spec.unpacker && runspec::XXPATCHER_ENUM == spec.patcher && spec.n_vecs == XXN_VEC && spec.n_vals == XXN_VAL) {
    XXCOLUMN_COPY
    kernels::device::alp::query_column<                                   
        T, XXN_VEC, XXN_VAL, 
        AlpUnpacker<T, XXN_VEC, XXN_VAL,
        XXUNPACKER_T,
            XXPATCHER_T<T, XXN_VEC, XXN_VAL >, 
            XXCOLUMN_T<T>>, XXCOLUMN_T<T>>
        <<<n_blocks, n_threads>>>(                 
            out, column, magic_value);            
    transfer::destroy_alp_column(column);
}
"""


ALP_COLUMN = "AlpColumn"
ALP_COLUMN_EXTENDED = "AlpExtendedColumn"
ALP_COLUMN_COPY = "auto column = transfer::copy_alp_column_to_gpu(data);"
ALP_COLUMN_EXTENDED_COPY = (
    "auto column = transfer::copy_alp_extended_column_to_gpu(data);"
)


def packer(name: str, is_alp: bool, additional_param: str | None = None) -> str:
    return f'{name}<T, XXN_VEC, XXN_VAL, {"BPFunctor<T>" if not is_alp else "ALPFunctor<T, XXN_VEC>"} {", " + additional_param if additional_param else ""} {", consts::VALUES_PER_VECTOR" if is_alp else ""}>'


def get_fls_parameters(for_alp: bool):
    return [
        MultiParameter(
            [
                Parameter(
                    "XXUNPACKER_ENUM",
                    [
                        "NON_INTERLEAVED",
                        "STATELESS",
                        "STATEFUL_CACHE",
                        "STATEFUL_LOCAL_MEMORY_1",
                        "STATEFUL_LOCAL_MEMORY_2",
                        "STATEFUL_LOCAL_MEMORY_4",
                        "STATEFUL_REGISTER_1",
                        "STATEFUL_REGISTER_2",
                        "STATEFUL_REGISTER_4",
                        "STATEFUL_REGISTER_BRANCHLESS_1",
                        "STATEFUL_REGISTER_BRANCHLESS_2",
                        "STATEFUL_REGISTER_BRANCHLESS_4",
                        "STATELESS_BRANCHLESS",
                        "STATEFUL_BRANCHLESS",
                    ],
                ),
                Parameter(
                    "XXUNPACKER_T",
                    [
                        packer("BitUnpackerNonInterleaved", for_alp),
                        packer("BitUnpackerStateless", for_alp),
                        packer(
                            "BitUnpackerStateful", for_alp, "CacheLoader<T, XXN_VEC>"
                        ),
                        packer(
                            "BitUnpackerStateful",
                            for_alp,
                            "LocalMemoryLoader<T, XXN_VEC, 1>",
                        ),
                        packer(
                            "BitUnpackerStateful",
                            for_alp,
                            "LocalMemoryLoader<T, XXN_VEC, 2>",
                        ),
                        packer(
                            "BitUnpackerStateful",
                            for_alp,
                            "LocalMemoryLoader<T, XXN_VEC, 4>",
                        ),
                        packer(
                            "BitUnpackerStateful",
                            for_alp,
                            "RegisterLoader<T, XXN_VEC, 1>",
                        ),
                        packer(
                            "BitUnpackerStateful",
                            for_alp,
                            "RegisterLoader<T, XXN_VEC, 2>",
                        ),
                        packer(
                            "BitUnpackerStateful",
                            for_alp,
                            "RegisterLoader<T, XXN_VEC, 4>",
                        ),
                        packer(
                            "BitUnpackerStateful",
                            for_alp,
                            "RegisterBranchlessLoader<T, XXN_VEC, 1>",
                        ),
                        packer(
                            "BitUnpackerStateful",
                            for_alp,
                            "RegisterBranchlessLoader<T, XXN_VEC, 2>",
                        ),
                        packer(
                            "BitUnpackerStateful",
                            for_alp,
                            "RegisterBranchlessLoader<T, XXN_VEC, 4>",
                        ),
                        packer("BitUnpackerStatelessBranchless", for_alp),
                        packer("BitUnpackerStatefulBranchless", for_alp),
                    ],
                ),
            ]
        ),
        Parameter("XXN_VEC", [1, 4]),
        Parameter("XXN_VAL", [1, 4]),
    ]


FLS_PARAMETERS = get_fls_parameters(False)

ALP_PARAMETERS = [
    *get_fls_parameters(True),
    MultiParameter(
        [
            Parameter(
                "XXPATCHER_ENUM",
                [
                    "STATELESS_P",
                    "STATEFUL_P",
                    "NAIVE",
                    "NAIVE_BRANCHLESS",
                    "PREFETCH_POSITION",
                    "PREFETCH_ALL",
                    "PREFETCH_ALL_BRANCHLESS",
                ],
            ),
            Parameter(
                "XXPATCHER_T",
                [
                    "StatelessALPExceptionPatcher",
                    "StatefulALPExceptionPatcher",
                    "NaiveALPExceptionPatcher",
                    "NaiveBranchlessALPExceptionPatcher",
                    "PrefetchPositionALPExceptionPatcher",
                    "PrefetchAllALPExceptionPatcher",
                    "PrefetchAllBranchlessALPExceptionPatcher",
                ],
            ),
            Parameter(
                "XXCOLUMN_T",
                [
                    ALP_COLUMN,
                    ALP_COLUMN,
                    ALP_COLUMN_EXTENDED,
                    ALP_COLUMN_EXTENDED,
                    ALP_COLUMN_EXTENDED,
                    ALP_COLUMN_EXTENDED,
                    ALP_COLUMN_EXTENDED,
                ],
            ),
            Parameter(
                "XXCOLUMN_COPY",
                [
                    ALP_COLUMN_COPY,
                    ALP_COLUMN_COPY,
                    ALP_COLUMN_EXTENDED_COPY,
                    ALP_COLUMN_EXTENDED_COPY,
                    ALP_COLUMN_EXTENDED_COPY,
                    ALP_COLUMN_EXTENDED_COPY,
                    ALP_COLUMN_EXTENDED_COPY,
                ],
            ),
        ]
    ),
]


def insert_parameters(
    code: Code, input_parameters: list[Parameter | MultiParameter]
) -> Code:
    results = [code]

    multi_parameters = [
        MultiParameter([p]) if isinstance(p, Parameter) else p for p in input_parameters
    ]

    for mp in multi_parameters:
        n_params = len(mp.parameters)
        n_values = len(mp.parameters[0].values)

        len_pre_application = len(results)
        results *= n_values
        for p in range(n_params):
            for v in range(n_values):
                tag = mp.parameters[p].tag
                value = mp.parameters[p].values[v]

                for i in range(len_pre_application):
                    index = i + v * len_pre_application
                    results[index] = results[index].replace(tag, str(value))

    return Code("".join(results))


def main(args):
    code = FILE_HEADER
    code += (
        FLS_DECOMPRESS_COLUMN_FUNCTION_SIGNATURE
        + insert_parameters(Code(FLS_DECOMPRESS_COLUMN_IF_STATEMENT), FLS_PARAMETERS)
        + FUNCTION_FOOTER
    )
    code += (
        FLS_QUERY_COLUMN_FUNCTION_SIGNATURE
        + insert_parameters(Code(FLS_QUERY_COLUMN_IF_STATEMENT), FLS_PARAMETERS)
        + FUNCTION_FOOTER
    )
    code += (
        FLS_COMPUTE_COLUMN_FUNCTION_SIGNATURE
        + insert_parameters(Code(FLS_COMPUTE_COLUMN_IF_STATEMENT), FLS_PARAMETERS)
        + FUNCTION_FOOTER
    )
    code += (
        ALP_DECOMPRESS_COLUMN_FUNCTION_SIGNATURE
        + insert_parameters(Code(ALP_DECOMPRESS_COLUMN_IF_STATEMENT), ALP_PARAMETERS)
        + FUNCTION_FOOTER
    )
    code += (
        ALP_QUERY_COLUMN_FUNCTION_SIGNATURE
        + insert_parameters(Code(ALP_QUERY_COLUMN_IF_STATEMENT), ALP_PARAMETERS)
        + FUNCTION_FOOTER
    )
    code += FILE_FOOTER

    args.out.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "-o", "--out", type=argparse.FileType("w"), default=sys.stdout, help="Out file"
    )

    parser.add_argument(
        "-ll",
        "--logging-level",
        type=int,
        default=logging.INFO,
        choices=[logging.CRITICAL, logging.ERROR, logging.INFO, logging.DEBUG],
        help=f"logging level to use: {logging.CRITICAL}=CRITICAL, {logging.ERROR}=ERROR, {logging.INFO}=INFO, "
        + f"{logging.DEBUG}=DEBUG, higher number means less output",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.logging_level)  # filename='program.log',
    logging.info(
        f"Started {os.path.basename(sys.argv[0])} with the following args: {args}"
    )
    main(args)
