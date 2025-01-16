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


GRAPH_TYPES_CLI_OPTIONS = {
    "scatter-speed": GraphTypes.SCATTER_SPEED,
}

DEFAULT_COLORS = [
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:red",
    "tab:pink",
    "tab:gray",
    "tab:cyan",
]


class GraphConfiguration:
    y_axis_range: tuple[int | None, int | None]
    show_legend: bool
    output_name: str | None
    h_line: int | None

    def __init__(self, args: argparse.Namespace) -> None:
        self.y_axis_range = (0 if args.start_from_zero else None, args.y_axis_max_value)
        self.h_line = args.h_line
        self.show_legend = args.show_legend
        self.output_name = args.output_file_path


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

    colors_index = 0
    fig, ax = plt.subplots()
    for name, set_results in results.group_by("set_name", maintain_order=True):
        name = name[0]
        x = []
        y = []

        for _, function_results in set_results.group_by(
            "function_id", maintain_order=True
        ):
            x.append(function_results[column_name][0])
            y.append(function_results["execution_speed"].mean() / 1000)

        ax.scatter(x, y, s=20, c=DEFAULT_COLORS[colors_index], label=name)

        colors_index += 1

    if config.h_line:
        ax.axhline(y=config.h_line / 1000, color='r', linestyle='--', label='Baseline')
    ax.set_ylim(config.y_axis_range)
    plt.xlabel(pretty_name)
    plt.ylabel("Average execution speed (us)")

    if config.show_legend:
        plt.legend()
    output_graph(config.output_name)




def load_dataframes(filenames: str) -> pl.DataFrame:
    files: list[str] = [filename.strip() for filename in args.files.split(":")]

    results: list[pl.DataFrame] = []
    for filename in files:
        df = pl.read_csv(filename)
        df = df.with_columns(
            pl.lit(filename.split("/")[-1].split(".")[0]).alias("set_name")
        )
        results.append(df)

    return pl.concat(results)


def main(args):
    config = GraphConfiguration(args)
    df = load_dataframes(args.files)

    plot_options = {
        GraphTypes.SCATTER_SPEED: plot_scatter,
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
        "-yamv",
        "--y-axis-max-value",
        type=int,
        default=None,
        help="End the y axis at number",
    )
    parser.add_argument(
        "-hl",
        "--h-line",
        type=int,
        default=None,
        help="Horizontal, dashed line",
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
