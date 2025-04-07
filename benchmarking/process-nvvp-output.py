#!/usr/bin/env python3

from io import StringIO
import os
import sys

import argparse
import logging
import itertools

import inspect
import types
from pathlib import Path

import polars as pl


def directory_exists(path: str) -> bool:
    return Path(path).is_dir()


def get_processing_functions() -> list[tuple[str, types.FunctionType]]:
    current_module = inspect.getmodule(inspect.currentframe())
    functions = inspect.getmembers(current_module, inspect.isfunction)

    processing_function_prefix = "process_"
    processing_functions = filter(
        lambda x: x[0].startswith(processing_function_prefix), functions
    )
    stripped_prefixes_from_name = map(
        lambda x: (x[0].replace(processing_function_prefix, ""), x[1]),
        processing_functions,
    )
    return list(stripped_prefixes_from_name)


def get_all_files_in_dir(dir: str) -> list[str]:
    file_paths = []

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    return file_paths


def get_all_files_with_prefix_in_dir(dir: str, prefix: str) -> list[str]:
    return list(
        filter(lambda x: x.split("/")[-1].startswith(prefix), get_all_files_in_dir(dir))
    )


def parse_duration(line: str) -> int:
    """
    Parses the time, returns as nanoseconds
    """
    multiplier = None

    duration = line.split()[1]
    unit: str = duration[-2]
    if unit == "u":
        multiplier = 1000
    elif unit == "m":
        multiplier = 1000000
    else:
        raise Exception(f"Could not parse {duration}, unknown unit: {unit}.")

    value_float: float = float(duration[:-2])

    return int(value_float * multiplier)


def read_profiler_output_as_df(file: str) -> pl.DataFrame:
    lines = []

    with open(file, "r") as f:
        lines = f.readlines()

    kernel_lines = filter(lambda x: "void" in x, lines)
    durations = list(map(parse_duration, kernel_lines))

    return pl.DataFrame(
        {"kernel_index": list(range(0, len(durations))), "duration (ns)": durations}
    )


def convert_ffor_file_to_df(file: str) -> pl.DataFrame:
    params = os.path.basename(file).split("-")

    df = read_profiler_output_as_df(file).with_columns(
        pl.lit(params[0]).alias("encoding"),
        pl.lit(params[2]).alias("data_type"),
        pl.lit(params[3]).alias("kernel"),
        pl.lit(params[4]).alias("unpack_n_vectors"),
        pl.lit(params[5]).alias("unpack_n_values"),
        pl.lit(params[6]).alias("unpacker"),
        pl.Series("vbw", range(int(params[7]), int(params[8]) + 1)),
        pl.lit(params[9]).alias("n_vecs"),
    )

    return df


def convert_alp_file_to_df(file: str) -> pl.DataFrame:
    params = file.split("-")

    df = read_profiler_output_as_df(file).with_columns(
        pl.lit(params[0]).alias("encoding"),
        pl.lit(params[2]).alias("data_type"),
        pl.lit(params[3]).alias("kernel"),
        pl.lit(params[4]).alias("unpack_n_vectors"),
        pl.lit(params[5]).alias("unpack_n_values"),
        pl.lit(params[6]).alias("unpacker"),
        pl.lit(params[7]).alias("patcher"),
        pl.lit(params[8]).alias("vbw"),
        pl.Series("ec", range(int(params[10]), int(params[11]))),
        pl.lit(params[12]).alias("n_vecs"),
    )

    return df


def collect_files_into_df(
    input_dir: str, prefix: str, convertor_lambda
) -> pl.DataFrame:
    alp_micro_files = get_all_files_with_prefix_in_dir(input_dir, prefix)
    return pl.concat(map(convertor_lambda, alp_micro_files))


def process_ffor_micro(input_dir: str) -> tuple[str, pl.DataFrame]:
    return "ffor-micro.csv", collect_files_into_df(
        input_dir, "ffor-micro", convert_ffor_file_to_df
    )


def process_alp_micro(input_dir: str) -> tuple[str, pl.DataFrame]:
    return "alp-micro.csv", collect_files_into_df(
        input_dir, "alp-micro", convert_alp_file_to_df
    )


def main(args):
    assert directory_exists(args.input_dir)
    assert directory_exists(args.output_dir)

    for default_name, df in args.processing_function(args.input_dir):
        df.write_csv(os.path.join(args.output_dir, default_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    processing_functions = {func[0]: func[1] for func in get_processing_functions()}
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "processing_function",
        type=str,
        choices=list(processing_functions.keys()) + ["all"],
        help="function to execute",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="directory_to_read_results_from",
    )
    parser.add_argument(
        "output_dir",
        default=None,
        type=str,
        help="directory_to_write_results_to",
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

    if args.processing_function == "all":
        args.processing_function = lambda in_dir: list(
            func(in_dir) for func in processing_functions.values()
        )
    else:
        string_value = args.processing_function
        args.processing_function = lambda in_dir: [
            processing_functions[string_value](in_dir)
        ]
    main(args)
