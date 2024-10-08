#!/usr/bin/python3

import os
import sys

import argparse
import logging
import enum
from pandas._libs.tslibs.fields import get_start_end_field

import polars as pl
import matplotlib.pyplot as plt
import dataclasses


class GraphTypes(enum.Enum):
    SCATTER_SPEED = 1
    SCATTER_AVG_SPEED = 2
    BOXPLOT_SPEED = 3
    BOXPLOT_SETS_SPEED = 4


GRAPH_TYPES_CLI_OPTIONS = {
    "scatter-speed": GraphTypes.SCATTER_SPEED,
    "scatter-average-speed": GraphTypes.SCATTER_AVG_SPEED,
    "boxplot-speed": GraphTypes.BOXPLOT_SPEED,
    "boxplot-sets-speed": GraphTypes.BOXPLOT_SETS_SPEED,
}

DEFAULT_COLORS = [
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:red",
    "tab:pink",
]


@dataclasses.dataclass
class GraphConfiguration:
    start_from_zero: bool
    show_legend: bool
    output_name: str | None


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


def convert_unique_str_to_colors(values: list[str]) -> list[str]:
    unique = list(set(values))
    return [DEFAULT_COLORS[unique.index(x)] for x in values]


def plot_scatter(results: pl.DataFrame, config: GraphConfiguration):
    column_name, pretty_name = return_default_x(results)
    sets = get_sets(results)
    x = [results[column_name]]
    y = [result / 1000 for result in results["execution_speed"]]
    colors = [sets.index(s) for s in results["set_name"]]

    plt.xlabel(pretty_name)
    plt.ylabel("Execution speed (us)")

    plt.scatter(x, y, s=20, c=colors)

    if config.show_legend:
        plt.legend()
    output_graph(config.output_name)


def plot_scatter_average(results: pl.DataFrame, config: GraphConfiguration):
    column_name, pretty_name = return_default_x(results)

    colors_index = 0
    for name, set_results in results.group_by("set_name", maintain_order=True):
        name = name[0]
        x = []
        y = []

        for _, function_results in set_results.group_by(
            "function_id", maintain_order=True
        ):
            x.append(function_results[column_name][0])
            y.append(function_results["execution_speed"].mean())

        plt.scatter(x, y, s=20, c=DEFAULT_COLORS[colors_index], label=name)
        colors_index += 1

    plt.xlabel(pretty_name)
    plt.ylabel("Execution speed (us)")

    if config.show_legend:
        plt.legend()
    output_graph(config.output_name)


def plot_boxplot(results: pl.DataFrame, config: GraphConfiguration):
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

    if config.start_from_zero:
        ax.set_ylim(ymin=0)

    output_graph(config.output_name)


def plot_boxplot_sets(results: pl.DataFrame, config: GraphConfiguration):
    x = []
    y = []

    for set_name, function_results in results.group_by("set_name", maintain_order=True):
        x.append(set_name[0])
        y.append([result / 1000 for result in function_results["execution_speed"]])

    fig, ax = plt.subplots()
    ax.boxplot(y)
    ax.set_ylabel("Execution speed (us)")
    ax.set_xticklabels(x)

    if config.start_from_zero:
        ax.set_ylim(ymin=0)

    output_graph(config.output_name)


def load_dataframes(filenames: str) -> pl.DataFrame:
    files: list[str] = [filename.strip() for filename in args.files.split(":")]

    results: list[pl.DataFrame] = []
    for filename in files:
        df = pl.read_csv(filename)
        df = df.with_columns(pl.lit(filename.split(".")[0]).alias("set_name"))
        results.append(df)

    return pl.concat(results)


def main(args):
    config = GraphConfiguration(
        args.start_from_zero, args.show_legend, args.output_file_path
    )
    df = load_dataframes(args.files)

    plot_options = {
        GraphTypes.SCATTER_SPEED: plot_scatter,
        GraphTypes.SCATTER_AVG_SPEED: plot_scatter_average,
        GraphTypes.BOXPLOT_SPEED: plot_boxplot,
        GraphTypes.BOXPLOT_SETS_SPEED: plot_boxplot_sets,
    }

    plot_options[args.plot](df, config)


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
        "-sl",
        "--show-legend",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Adds a legend to the graph",
    )
    parser.add_argument(
        "-sfz",
        "--start-from-zero",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Start the y axis from zero",
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
