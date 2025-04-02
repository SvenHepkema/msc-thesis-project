#!/usr/bin/python3

import os
import sys

import argparse
import logging

GENERATED_BINDINGS_DIR = "./src/generated-bindings/"

FILE_HEADER = """
#include <stdexcept>

#include "kernel-bindings.cuh"
#include "../engine/kernels.cuh"

namespace bindings{
"""

FILE_FOOTER = """
}
"""


DATA_TYPES = [
    "uint32_t",
    "uint64_t",
    "float",
    "double",
]

FUNCTIONS = [
    "decompress_column",
    "query_column",
    "compute_column",
    "query_multi_column",
]

ENCODINGS = [
    "BP",
    "FFOR",
    "ALP",
    "ALPExtended",
]

UNPACKERS = [
    "Stateless",
    "StatelessBranchless",
    "StatefulCache",
    "StatefulLocal1",
    "StatefulLocal2",
    "StatefulLocal4",
    "StatefulRegister1",
    "StatefulRegister2",
    "StatefulRegister4",
    "StatefulRegisterBranchless1",
    "StatefulRegisterBranchless2",
    "StatefulRegisterBranchless4",
    "StatefulBranchless",
]
BEST_UNPACKER = [
    "StatefulBranchless",
]


PATCHERS = [
    "None",
    "Stateless",
    "Stateful",
    "Naive",
    "NaiveBranchless",
    "PrefetchPosition",
    "PrefetchAll",
    "PrefetchAllBranchless",
]
BEST_PATCHER = [
    "PrefetchAllBranchless",
]


def get_column_t(encoding: str, data_type: str) -> str:
    column_t = f"BPColumn<{data_type}>"
    if "FFOR" in encoding:
        column_t = f"FFORColumn<{data_type}>"
    elif "ALPExtended" in encoding:
        column_t = f"ALPExtendedColumn<{data_type}>"
    elif "ALP" in encoding:
        column_t = f"ALPColumn<{data_type}>"
    return "flsgpu::device::" + column_t


def get_decompressor_type(
    encoding: str,
    data_type: str,
    unpacker: str,
    patcher: str,
    n_vec: int,
    n_val: int,
) -> str:
    column_t = get_column_t(encoding, data_type)
    functor = f"BPFunctor<{data_type}>"
    patcher_t = f""
    decompressor_t = f"{encoding}Decompressor"
    if "FFOR" in encoding:
        functor = f"FFORFunctor<{data_type}, {n_vec}>"
    elif "ALPExtended" in encoding:
        functor = f"ALPFunctor<{data_type}, {n_vec}>"
        patcher_t = f"flsgpu::device::{patcher}ALPExceptionPatcher<{data_type}, {n_vec}, {n_val}>,"
        decompressor_t = "ALPDecompressor"
    elif "ALP" in encoding:
        functor = f"ALPFunctor<{data_type}, {n_vec}>"
        patcher_t = f"flsgpu::device::{patcher}ALPExceptionPatcher<{data_type}, {n_vec}, {n_val}>,"

    loader_t = ""
    if "Stateful" in unpacker and "StatefulBranchless" not in unpacker:
        loader_t = ", flsgpu::device::"
        if "Cache" in unpacker:
            loader_t += f"CacheLoader<{data_type}, {n_vec}>"
        elif "Local" in unpacker:
            loader_t += f"LocalMemoryLoader<{data_type}, {n_vec}, {unpacker[-1]}>"
        elif "RegisterBranchless" in unpacker:
            loader_t += (
                f"RegisterBranchlessLoader<{data_type}, {n_vec}, {unpacker[-1]}>"
            )
        elif "Register" in unpacker:
            loader_t += f"RegisterLoader<{data_type}, {n_vec}, {unpacker[-1]}>"
        unpacker = "Stateful"

    unpacker_t = f"flsgpu::device::BitUnpacker{unpacker}<{data_type}, {n_vec}, {n_val},  flsgpu::device::{functor} {loader_t}>"

    return f"flsgpu::device::{decompressor_t}<{data_type}, {n_vec}, {unpacker_t}, {patcher_t} {column_t}>"


def get_if_statement(
    encoding: str,
    data_type: str,
    function: str,
    n_vec: int,
    n_val: int,
    unpacker: str,
    patcher: str,
    n_columns: int | None = None,
    n_repetitions: int | None = None,
) -> str:
    assert data_type in DATA_TYPES
    assert function in FUNCTIONS
    assert n_vec in [1, 2, 4, 8]
    assert n_val in [1, 32]
    assert unpacker in UNPACKERS
    assert patcher in PATCHERS

    column_t = get_column_t(encoding, data_type)
    decompressor_t = get_decompressor_type(
        encoding, data_type, unpacker, patcher, n_vec, n_val
    )

    return (
        f"if (unpack_n_vectors == {n_vec} && unpack_n_values == {n_val} && unpacker == Unpacker::{unpacker} && patcher == Patcher::{patcher} {'&& n_columns == ' + str(n_columns) if n_columns else ''}) "
        + "{"  # }
        f"return kernels::host::{function}<{data_type}, {n_vec}, {n_val}, {decompressor_t}, {column_t} {',' + str(n_repetitions) if n_repetitions else ''}>(column);"
        "}"
    )


def get_function(
    encoding: str,
    data_type: str,
    name: str,
    return_type: str,
    content: list[str],
    is_multi_column: bool = False,
    is_compute_column: bool = False,
) -> str:
    assert not (is_multi_column and is_compute_column)
    column_t = get_column_t(encoding, data_type)
    return (
        f"{return_type} {name}(const {column_t} column, const unsigned unpack_n_vectors, const unsigned unpack_n_values, const Unpacker unpacker, const Patcher patcher {', const unsigned n_columns' if is_multi_column else ''}{', const unsigned n_repetitions' if is_compute_column else ''})"
        + "{"
        + "\n".join(content)
        + f'throw std::invalid_argument("Could not find correct binding in {name} {encoding}<{data_type}>");'
        + "}"
    )


def write_file(
    file_name: str,
    functions: list[str],
):
    logging.info(f"Writing file {file_name}")
    with open(os.path.join(GENERATED_BINDINGS_DIR, file_name), "w") as f:
        f.write("\n".join([FILE_HEADER] + functions + [FILE_FOOTER]))


def main(args):
    for encoding in ["BP", "FFOR"]:
        for data_type in ["uint32_t", "uint64_t"]:
            for binding, return_type in zip(
                ["decompress_column", "query_column"], [None, "bool"]
            ):
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            return_type if return_type else data_type + "*",
                            [
                                get_if_statement(
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    unpacker,
                                    "None",
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for unpacker in UNPACKERS
                            ],
                        )
                    ],
                )

    for encoding in ["FFOR"]:
        for data_type in ["uint32_t", "uint64_t"]:
            for binding, options in zip(
                # Reinsert when multcolumn is done
                # ["query_multi_column", "compute_column"],
                # [(True, False), (False, True)]
                ["compute_column"],
                [(False, True)],
            ):
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            "bool",
                            [
                                get_if_statement(
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    unpacker,
                                    "None",
                                    n_repetitions=10
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for unpacker in UNPACKERS
                            ],
                            is_multi_column=options[0],
                            is_compute_column=options[1],
                        )
                    ],
                )

    for encoding, patchers_per_encoding in zip(
        ["ALP", "ALPExtended"], [PATCHERS[1:3], PATCHERS[3:]]
    ):
        for data_type in ["float", "double"]:
            for binding, option, return_type, unpackers, patchers in zip(
                # Reinsert when multcolumn is done
                # ["decompress_column", "query_column", "query_multi_column"],
                # [False, False, True],
                #[None, "bool", "bool"],
                # [UNPACKERS, UNPACKERS, BEST_UNPACKER],
                # [patchers_per_encoding, patchers_per_encoding, BEST_PATCHER],
                ["decompress_column", "query_column"],
                [False, False],
                [None, "bool"],
                [UNPACKERS, UNPACKERS],
                [patchers_per_encoding, patchers_per_encoding],
            ):
                n_cols = range(1, 10 + 1) if option else [None]
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            return_type if return_type else data_type + "*",
                            [
                                get_if_statement(
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    unpacker,
                                    patcher,
                                    n_columns=n_col,
                                    n_repetitions=None,
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for n_col in n_cols
                                for unpacker in unpackers
                                for patcher in patchers
                            ],
                            is_multi_column=option,
                        )
                    ],
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

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
