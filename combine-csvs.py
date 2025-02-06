#!/usr/bin/python3

import os
import sys

import polars as pl
import argparse
import logging
import io
from typing import Generator

OUTPUT_FILE_SECTION_BOUNDARY = "=DATA="


def open_files_in_dir(
    dir_path: str, extension: str | None = None
) -> Generator[tuple[str, io.TextIOWrapper], None, None]:
    for file_str in os.listdir(os.fsencode(dir_path)):
        file_name = os.fsdecode(file_str)
        if extension is None or file_name.endswith(extension):
            with open(os.path.join(dir_path, file_name), "r") as file:
                yield file_name, file


def process_command(command: str) -> dict[str, str]:
    values = command.split(" ")

    return {
        "experiment": values[1],
        "unpacker": values[2],
        "patcher": values[3],
        "n_vecs": values[4],
        "n_vals": values[5],
        "datatype_width": values[6],
        "data_source": values[7],
        "data_generation": values[8],
        "n_vec_count": values[9],
    }


def process_file(content: str) -> pl.DataFrame | None:
    split_content = content.split(OUTPUT_FILE_SECTION_BOUNDARY)

    if len(split_content) != 2:
        return None

    process, csv_str = split_content
    df = pl.read_csv(io.StringIO(csv_str))

    process_lines = process.split("\n")
    command = process_command(process_lines[0])
    command["n_failed_runs"] = str(len(process.split("\n")) - 2)

    for name, value in reversed(command.items()):
        df.insert_column(0, pl.Series(name, [value] * df.height))

    return df


def main(args):
    dfs = []
    for filename, file in open_files_in_dir(args.dir):
        result = process_file(file.read())

        if result is None:
            logging.warning(f"Failed to process {filename}")
        else:
            dfs.append(result)

    df: pl.DataFrame = pl.concat(dfs)
    df.write_csv(args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument("dir", type=str, help="Directory containing data files")
    parser.add_argument(
        "-e", "--extension", type=str, help="Extension of files to filter"
    )
    parser.add_argument(
        "-o",
        "--out",
        default=sys.stdout,
        type=argparse.FileType("w"),
        help="Output file",
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
