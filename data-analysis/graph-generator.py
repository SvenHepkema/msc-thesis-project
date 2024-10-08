#!/usr/bin/python3

import os
import sys

import argparse
import logging
import enum

import polars as pl
import matplotlib.pyplot as plt


class GraphTypes(enum.Enum):
    SCATTER_SPEED = 1
    BOXPLOT_SPEED = 2


GRAPH_TYPES_CLI_OPTIONS = {
    "scatter-speed": GraphTypes.SCATTER_SPEED,
    "boxplot-speed": GraphTypes.BOXPLOT_SPEED,
}


def return_default_x(df: pl.DataFrame) -> tuple[str, str]:
    names = [
        ("value_bit_width", "Value bit widths"),
        ("exception_count", "Exception counts"),
    ]

    for column_name, pretty_name in names:
        if column_name in df:
            return column_name, pretty_name

    return "id", "Function ID"

def output_graph(output_name: str|None=None):
    if output_name is None:
        plt.show()
    elif output_name[-4:] == "eps":
        plt.savefig(output_name, format='eps')
    else: 
        plt.savefig(output_name, dpi=1000, format='png', bbox_inches='tight')

def plot_scatter(results: pl.DataFrame, output_name: str|None=None):
    column_name, pretty_name = return_default_x(results)
    x = [results[column_name]]
    y = [result / 1000 for result in results["execution_speed"]]

    plt.xlabel(pretty_name)
    plt.ylabel("Execution speed (us)")

    plt.scatter(x, y, s=20)

    output_graph(output_name)


def plot_boxplot(results: pl.DataFrame, output_name: str|None=None):
    column_name, pretty_name = return_default_x(results)
    x = []
    y = []

    for _, function_results in results.group_by("function_id", maintain_order=True):
        i = function_results[column_name][0]
        x.append(i if i % 5 == 0 else "")
        y.append([result / 1000 for result in function_results["execution_speed"]])

    fig, ax = plt.subplots()
    ax.boxplot(y)
    ax.set_xlabel(pretty_name)
    ax.set_ylabel("Execution speed (us)")
    ax.set_xticklabels(x)

    output_graph(output_name)



def main(args):
    results = pl.read_csv(args.csv)

    plot_options = {
        GraphTypes.SCATTER_SPEED: plot_scatter,
        GraphTypes.BOXPLOT_SPEED: plot_boxplot,
    }

    plot_options[args.plot](results, args.output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument("csv", type=str, help="Input csv with results.")
    parser.add_argument(
        "plot",
        type=str,
        choices=GRAPH_TYPES_CLI_OPTIONS.keys(),
        help=f"Specify how to plot the data.",
    )
    parser.add_argument(
        "-o",
        "--output-file-path",
        type=str,
        default=None,
        help=f"Filename to save graph to",
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
