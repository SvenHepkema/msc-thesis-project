#!/usr/bin/python3

import os
import sys

import argparse
import logging
import enum
from typing import NewType
from io import StringIO

import polars as pl
import matplotlib.pyplot as plt


class GraphTypes(enum.Enum):
    SCATTER = 1
    BAR = 2


GRAPH_TYPES_CLI_OPTIONS = {
    "scatter": GraphTypes.SCATTER,
    "bar": GraphTypes.BAR,
}

COLOR_SET = [
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:red",
    "tab:pink",
    "tab:gray",
    "tab:cyan",
]

LEGEND_POSITIONS = [
    "best",
    "upper-right",
    "upper-left",
    "lower-left",
    "lower-right",
    "right",
    "center-left",
    "center-right",
    "lower-center",
    "upper-center",
    "center",
]

COLUMN_ALIASES_FOR_AXIS = {
    "execution_time": "Median execution time (us)",
    "ipc": "Instructions executed per cycle",
    "inst_executed_global_loads": "Global loads executed",
    "dram_read_bytes": "Total bytes read from DRAM",
    "dram_write_bytes": "Total bytes written to DRAM",
    "global_hit_rate": "L1 Cache global load hit rate",
    "eligible_warps_per_cycle": "Average number of eligible warps / active cycle",
}

COLUMN_ALIASES_FOR_TITLE  = {
    "execution_time": "Execution time",
    "ipc": "Instructions executed per cycle",
    "inst_executed_global_loads": "Global loads executed",
    "dram_read_bytes": "Total bytes read from DRAM",
    "dram_write_bytes": "Total bytes written to DRAM",
    "global_hit_rate": "L1 Cache global load hit rate",
    "eligible_warps_per_cycle": "Average number of eligible warps / active cycle",
}


ColumnName = NewType("ColumnName", str)


class GraphConfiguration:
    x_axis_range: tuple[int | None, int | None]
    y_axis_range: tuple[int | None, int | None]
    show_legend: bool
    legend_position: str
    output_name: str | None
    title: str | None
    h_line: int | None

    def __init__(self, args: argparse.Namespace) -> None:
        self.x_axis_range = (0 if args.start_from_zero else None, None)
        self.y_axis_range = (0 if args.start_from_zero else None, args.y_axis_max_value)
        self.h_line = args.h_line
        self.show_legend = args.show_legend
        self.legend_position = args.legend_position
        self.output_name = args.output_file_path
        self.title = args.title

    def apply_to_ax(self, ax):
        if self.h_line:
            ax.axhline(
                y=self.h_line / 1000, color="r", linestyle="--", label="Baseline"
            )

        #ax.set_xticks(range(0, 35, 5))  
        #ax.set_xticks(range(32), minor=True)
        #ax.tick_params(axis='x', which='minor', length=4, color='r', labelbottom=False)  # Minor ticks in red, no labels

        ax.grid(which='both', linestyle='--', linewidth=0.1)
        ax.set_xlim(self.x_axis_range)
        ax.set_ylim(self.y_axis_range)

        if self.show_legend:
            ax.legend(scatterpoints=1, fontsize=12, loc=self.legend_position.replace("-", " "))



    def output_graph(self):
        plt.tight_layout()
        if self.output_name is None:
            plt.show()
        elif self.output_name[-4:] == "eps":
            plt.savefig(self.output_name, format="eps")
        else:
            plt.savefig(self.output_name, dpi=1000, format="png", bbox_inches="tight")


class ResultsFile:
    name: str
    command: str
    data: pl.DataFrame

    def __init__(self, filename: str) -> None:
        self.name = filename
        with open(filename) as file:
            lines = [line for line in file]
            self.command = lines[0]
            csv = "".join(lines[1:])
            self.data = pl.read_csv(StringIO(csv))


def load_files(file_names_arg: str) -> list[ResultsFile]:
    return [ResultsFile(file_name) for file_name in file_names_arg.split(":")]


def parse_column_names(
    column_names_arg: str, files: list[ResultsFile]
) -> list[ColumnName]:
    inner_set_columns = set()

    for file in files:
        for column_name in file.data.columns:
            inner_set_columns.add(column_name)

    result = []
    for column_name in column_names_arg.split(":"):
        if column_name in inner_set_columns:
            result.append(ColumnName(column_name))
        else:
            exit(
                f"Specified column name was not found in every dataset: "
                f"{column_name}, possible columnnames: {inner_set_columns}"
            )

    return result


def get_default_x_axis_column(df: pl.DataFrame) -> tuple[str, str]:
    names = [
        ("vbw", "Value bit widths"),
        ("ec", "Exception counts"),
    ]

    for column_name, pretty_name in names:
        if column_name in df:
            return column_name, pretty_name

    return "id", "Function ID"


def plot_scatter(
    datasets: list[ResultsFile], columns: list[ColumnName], config: GraphConfiguration
):
    plt.style.use("classic")
    fig, ax = plt.subplots()

    x_axis_column, pretty_x_axis_name = get_default_x_axis_column(datasets[0].data)

    for column in columns:
        for i, dataset in enumerate(datasets):
            x = dataset.data[x_axis_column]
            y = dataset.data[column]

            if columns[0] == "execution_time":
                y = [value / 1000 for value in y]
            ax.scatter(
                x,
                y,
                s=24,
                linewidths=0,
                c=COLOR_SET[i],
                label=dataset.name,
            )

        ax.set_xlabel(pretty_x_axis_name)
        ax.set_ylabel(COLUMN_ALIASES_FOR_AXIS.get(column, column))
        ax.set_title(COLUMN_ALIASES_FOR_TITLE[column], fontsize=20)
        config.apply_to_ax(ax)

    config.output_graph()


def plot_bar(
    data: list[ResultsFile], columns: list[ColumnName], config: GraphConfiguration
):
    pass


PLOT_FUNCTION_MAPPING = {
    GraphTypes.SCATTER: plot_scatter,
    GraphTypes.BAR: plot_bar,
}


def main(args):
    results = load_files(args.files)
    assert len(results) > 0

    columns = parse_column_names(args.columns, results)
    assert len(columns) > 0

    config = GraphConfiguration(args)

    PLOT_FUNCTION_MAPPING[args.plot](results, columns, config)


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
        "columns",
        type=str,
        help=f"Specify which column to plot.",
    )
    parser.add_argument(
        "-o",
        "--output-file-path",
        type=str,
        default=None,
        help=f"Filename to save graph to",
    )

    # Graph styling
    parser.add_argument(
        "-sl",
        "--show-legend",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Adds a legend to the graph",
    )
    parser.add_argument(
        "-lp",
        "--legend-position",
        type=str,
        choices=LEGEND_POSITIONS,
        default=LEGEND_POSITIONS[0],
        help="Defines the position of the legend",
    )
    parser.add_argument(
        "-t",
        "--title",
        type=str,
        default=None,
        help="Title of graph",
    )
    parser.add_argument(
        "-sfz",
        "--start-from-zero",
        type=bool,
        default=True,
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
