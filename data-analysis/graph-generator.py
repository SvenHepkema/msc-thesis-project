#!/usr/bin/python3

import os
import sys

import argparse
import logging
import enum
from pandas._libs.tslibs.fields import get_start_end_field

import polars as pl
import matplotlib.pyplot as plt


class GraphTypes(enum.Enum):
    SCATTER_SPEED = 1
    SCATTER_AVG_SPEED = 2
    BOXPLOT_SPEED = 3


GRAPH_TYPES_CLI_OPTIONS = {
    "scatter-speed": GraphTypes.SCATTER_SPEED,
    "scatter-average-speed": GraphTypes.SCATTER_AVG_SPEED,
    "boxplot-speed": GraphTypes.BOXPLOT_SPEED,
}

DEFAULT_COLORS = [
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:red",
    "tab:pink",
]


def return_default_x(df: pl.DataFrame) -> tuple[str, str]:
    names = [
        ("value_bit_width", "Value bit widths"),
        ("exception_count", "Exception counts"),
    ]

    for column_name, pretty_name in names:
        if column_name in df:
            return column_name, pretty_name

    return "id", "Function ID"


def get_sets(df: pl.DataFrame) -> list[str]:
    return list(df.unique(subset=["set_name"])["set_name"])


def output_graph(output_name: str | None = None):
    if output_name is None:
        plt.show()
    elif output_name[-4:] == "eps":
        plt.savefig(output_name, format="eps")
    else:
        plt.savefig(output_name, dpi=1000, format="png", bbox_inches="tight")


def convert_unique_str_to_colors(values: list[str])-> list[str]:
    unique = list(set(values))
    return [DEFAULT_COLORS[unique.index(x)] for x in values]


def plot_scatter(results: pl.DataFrame, legend: bool, output_name: str | None = None):
    column_name, pretty_name = return_default_x(results)
    sets = get_sets(results)
    x = [results[column_name]]
    y = [result / 1000 for result in results["execution_speed"]]
    colors = [sets.index(s) for s in results["set_name"]]

    plt.xlabel(pretty_name)
    plt.ylabel("Execution speed (us)")

    plt.scatter(x, y, s=20, c=colors)

    if legend:
        plt.legend()
    output_graph(output_name)


def plot_scatter_average(results: pl.DataFrame, legend: bool, output_name: str | None = None):
    column_name, pretty_name = return_default_x(results)

    colors_index = 0
    for name, set_results in results.group_by("set_name", maintain_order=True):
        name = name[0]
        x = []
        y = []

        for _, function_results in set_results.group_by("function_id", maintain_order=True):
            x.append(function_results[column_name][0])
            y.append(function_results["execution_speed"].mean())

        plt.scatter(x, y, s=20, c=DEFAULT_COLORS[colors_index], label=name)
        colors_index += 1

    plt.xlabel(pretty_name)
    plt.ylabel("Execution speed (us)")

    if legend:
        plt.legend()
    output_graph(output_name)


def plot_boxplot(results: pl.DataFrame, legend: bool, output_name: str | None = None):
    column_name, pretty_name = return_default_x(results)

    assert len(get_sets(results)) == 1, "Boxplot only allows one set"

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

    if legend:
        plt.legend()
    output_graph(output_name)


def load_dataframes(filenames: str) -> pl.DataFrame:
    files: list[str] = [filename.strip() for filename in args.files.split(":")]

    results: list[pl.DataFrame] = []
    for filename in files:
        df = pl.read_csv(filename)
        df = df.with_columns(pl.lit(filename.split(".")[0]).alias("set_name"))
        results.append(df)

    return pl.concat(results)


def main(args):
    df = load_dataframes(args.files)

    plot_options = {
        GraphTypes.SCATTER_SPEED: plot_scatter,
        GraphTypes.SCATTER_AVG_SPEED: plot_scatter_average,
        GraphTypes.BOXPLOT_SPEED: plot_boxplot,
    }

    plot_options[args.plot](df, args.legend, args.output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "files", type=str, help="Input files with results. Delimited by ':'"
    )
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
        "-l",
        "--legend",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Adds a legend to the graph",
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
