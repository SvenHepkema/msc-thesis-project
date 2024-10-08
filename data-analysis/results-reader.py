#!/usr/bin/python3

import os
import sys

import argparse
import logging
import dataclasses
import enum

import polars as pl

nanoseconds = int

class Variation(enum.Enum):
    Generic = 1
    ValueBitWidth = 2
    ExceptionCount = 3

VARIATION_MAPPING = {
    "generic": Variation.Generic,
    "value-bit-width": Variation.ValueBitWidth,
    "exception-count": Variation.ExceptionCount,
}

@dataclasses.dataclass
class ExecutionRecord:
    id: int
    time: nanoseconds


@dataclasses.dataclass
class FunctionRecord:
    id: int
    runs: list[ExecutionRecord]

    def __repr__(self) -> str:
        return f"FunctionRecord({len(self.runs)} records)"

Results = list[FunctionRecord]


def parse_nvprof_line(i: int, line: str) -> ExecutionRecord:
    line = line.strip()
    multiplier = None

    unit: str = line[-2]
    if unit == "u":
        multiplier = 1000
    elif unit == "m":
        multiplier = 1000000
    else:
        raise Exception(f"Could not parse {line}, unknown unit: {unit}.")

    value: float = float(line[:-2])

    return ExecutionRecord(i, int(value * multiplier))


def read_results_file(
        file_path: str,  repeat: int | None = None
) -> Results:
    logging.info(f"Opening file {file_path}")


    # records = [[] for _ in range(n_unique_calls)]
    records = None
    with open(file_path) as file:
        records = [parse_nvprof_line(i, line) for i, line in enumerate(file)]

        logging.info(f"Read file {file_path}. Found {len(records)} records.")

    if repeat is None:
        return [FunctionRecord(i, [record]) for i, record in enumerate(records)]

    n_unique_calls = int(len(records) / repeat)
    records_per_function = [[] for _ in range(n_unique_calls)]
    for i, record in enumerate(records):
        records_per_function[i % n_unique_calls].append(record)

    return [FunctionRecord(i, record_set) for i, record_set in enumerate(records_per_function)]

def convert_results_to_dataframe(results: Results, variation: Variation) -> pl.DataFrame:
    logging.info(f"Creating dataframe with variation {variation}")
    dataframe = {
            "id": [],
            "function_id": [],
            "execution_speed": [],
            }

    if variation == Variation.ValueBitWidth:
        dataframe["value_bit_width"] = []
    elif variation == Variation.ExceptionCount:
        dataframe["exception_count"] = []

    for function in results:
        function_id = function.id
        for run in function.runs:
            dataframe["id"].append(run.id)
            dataframe["function_id"].append(function_id)
            dataframe["execution_speed"].append(run.time)

            if variation == Variation.ValueBitWidth:
                dataframe["value_bit_width"].append(function_id)
            elif variation == Variation.ExceptionCount:
                dataframe["exception_count"].append(function_id)

    logging.info("Created dataframe")
    return pl.DataFrame(dataframe)

def main(args):
    results = read_results_file(args.results_file,  args.repeat)
    df = convert_results_to_dataframe(results, args.variation)
    logging.info("Writing results to csv")
    df.write_csv(args.output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument("results_file", type=str, help="Input file with result times.")
    parser.add_argument("output_filename", type=str, help="Output file name of csv.")
    parser.add_argument(
        "-v",
        "--variation",
        type=str,
        default=next(iter(VARIATION_MAPPING)),
        choices=VARIATION_MAPPING.keys(),
        help=f"Variation in runs used, choices are: {VARIATION_MAPPING.keys()}"
    )
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=None,
        help="Specify how often the experiment was repeated for each function call.",
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
    args.variation = VARIATION_MAPPING[args.variation]
    logging.basicConfig(level=args.logging_level)  # filename='program.log',
    logging.info(
        f"Started {os.path.basename(sys.argv[0])} with the following args: {args}"
    )
    main(args)
