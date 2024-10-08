#!/usr/bin/python3

import os
import sys

import argparse
import logging
import dataclasses
import enum

import matplotlib.pyplot as plt

nanoseconds = int


class GraphTypes(enum.Enum):
    SCATTER = 1
    BOXPLOT = 2


GRAPH_TYPES_CLI_OPTIONS = {
    "scatter-execution": GraphTypes.SCATTER,
    "boxplot-execution": GraphTypes.BOXPLOT,
}

X_AXIS_CLI_OPTIONS = {
        "exception-count": "Exceptions per vector (1024)",
        "value-bit-width": "Value bit width",
}

@dataclasses.dataclass
class ExecutionRecord:
    time: nanoseconds


@dataclasses.dataclass
class FunctionRecord:
    runs: list[ExecutionRecord]

    def __repr__(self) -> str:
        return f"FunctionRecord({len(self.runs)} records)"

Results = list[FunctionRecord]


def parse_nvprof_line(line: str) -> ExecutionRecord:
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

    return ExecutionRecord(int(value * multiplier))


def read_results_file(
    file_path: str, repeat: int | None = None
) -> Results:
    logging.info(f"Opening file {file_path}")

    # records = [[] for _ in range(n_unique_calls)]
    records = None
    with open(file_path) as file:
        records = [parse_nvprof_line(line) for line in file]

        logging.info(f"Read file {file_path}. Found {len(records)} records.")

    if repeat is None:
        return [FunctionRecord([record]) for record in records]

    n_unique_calls = int(len(records) / repeat)
    records_per_function = [[] for _ in range(n_unique_calls)]
    for i, record in enumerate(records):
        records_per_function[i % n_unique_calls].append(record)

    return [FunctionRecord(record_set) for record_set in records_per_function]

def plot_scatter(results: Results):
    x = []
    y = []
    for i, function_result in enumerate(results):
        for run in function_result.runs:
            x.append(i)
            y.append(run.time)

    plt.scatter(x, y)
    plt.show()

def plot_boxplot(results: Results):
    x = []
    y = []
    for i, function_result in enumerate(results):
        x.append(i if i % 5 == 0 else '')
        y.append([run.time for run in function_result.runs])

    fig, ax = plt.subplots()
    ax.boxplot(y)
    ax.set_xticklabels(x)

    plt.show()

def main(args):
    results = read_results_file(args.results_file, args.repeat)

    plot_options = {
        GraphTypes.SCATTER: plot_scatter,
        GraphTypes.BOXPLOT: plot_boxplot,
    }

    plot_options[args.plot](results, x_axis_title=args.x_axis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument("results_file", type=str, help="Input file with result times.")
    parser.add_argument(
        "plot",
        type=str,
        choices=GRAPH_TYPES_CLI_OPTIONS.keys(),
        help=f"Specify how to plot the data. Choices: {GRAPH_TYPES_CLI_OPTIONS.keys()}",
    )
    parser.add_argument(
        "x-axis",
        type=str,
        choices=X_AXIS_CLI_OPTIONS.keys(),
        help=f"Specify what the x-axis is. Choices: {X_AXIS_CLI_OPTIONS.keys()}",
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
    args.plot = GRAPH_TYPES_CLI_OPTIONS[args.plot]
    logging.basicConfig(level=args.logging_level)  # filename='program.log',
    logging.info(
        f"Started {os.path.basename(sys.argv[0])} with the following args: {args}"
    )
    main(args)
