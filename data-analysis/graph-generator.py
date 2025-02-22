#!/usr/bin/python3

import os
import sys

import argparse
import logging
import enum
import math
from typing import Any, Callable, NewType, TextIO
from io import StringIO

import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


class GraphTypes(enum.Enum):
    SCATTER = 1
    BAR = 2


OUTPUT_FILE_SECTION_BOUNDARY = "=DATA="

GRAPH_TYPES_CLI_OPTIONS = {
    "scatter": GraphTypes.SCATTER,
    "bar": GraphTypes.BAR,
}

COLOR_SET = [
    "tab:blue",
    "tab:green",
    "tab:red",
    "tab:orange",
    "tab:pink",
    "tab:cyan",
    "tab:olive",
    "tab:gray",
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


class Metric:
    name: str
    column_alias: str
    title_alias: str

    y_max: int | None
    mutator: Callable[[Any], Any]
    formatter: Callable[[plt.Axes], None]

    def __init__(
        self,
        name: str,
        column_alias: str,
        title_alias: str | None = None,
        y_max: int | None = None,
        is_percentage: bool = False,
        mutator: Callable[[Any], Any] | None = None,
        formatter: Callable[[plt.Axes], None] | None = None,
    ) -> None:
        self.name = name
        self.column_alias = column_alias
        self.title_alias = title_alias if title_alias else column_alias

        if is_percentage:
            y_max = 100
            formatter = lambda ax: ax.yaxis.set_major_formatter(
                PercentFormatter(xmax=100)
            )

        self.y_max = y_max
        self.mutator = mutator if mutator else lambda x: x
        self.formatter = formatter if formatter else lambda _: None


BYTES_PER_KILOBYTE = 1024
CONVERT_BYTES_TO_KILOBYTES = lambda x: x / BYTES_PER_KILOBYTE

NVPROF_METRICS = [
    Metric(
        "execution_time",
        "Median execution time (us)",
        title_alias="Execution time",
        mutator=lambda x: x / 1000,
    ),
    Metric("ipc", "Instructions executed per cycle", y_max=4),
    Metric("inst_executed_global_loads", "Global loads executed"),
    Metric(
        "dram_read_bytes",
        "Total kilobytes read from DRAM",
        mutator=CONVERT_BYTES_TO_KILOBYTES,
    ),
    Metric(
        "dram_write_bytes",
        "Total kilobytes written to DRAM",
        mutator=CONVERT_BYTES_TO_KILOBYTES,
    ),
    Metric(
        "global_hit_rate",
        "L1 Cache global load hit rate",
        is_percentage=True,
    ),
    Metric(
        "l2_tex_hit_rate",
        "L2 Cache all request hit rate",
        is_percentage=True,
    ),
    Metric(
        "stall_memory_dependency",
        "% Stalls due to memory dependency",
        is_percentage=True,
    ),
    Metric(
        "stall_memory_throttle", "% Stalls due to memory throttle", is_percentage=True
    ),
    Metric("inst_issued", "Instructions issued"),
    Metric(
        "eligible_warps_per_cycle", "Average number of eligible warps / active cycle"
    ),
]
NVPROF_METRICS_MAPPING = {metric.name: metric for metric in NVPROF_METRICS}

def shift_list_to_right(values: list[Any]) -> list[Any]:
    return [values[-1]] + values[:-1]


ColumnName = NewType("ColumnName", str)


class GraphConfiguration:
    x_axis_range: tuple[int | None, int | None]
    y_axis_range: tuple[int | None, int | None]
    show_legend: bool
    legend_position: str
    legend_font_size: int
    output_name: str | None
    title: str | None
    show_subfigure_title: str | None
    h_line: int | None
    h_line_label: str | None

    def __init__(self, args: argparse.Namespace) -> None:
        self.x_axis_range = (0 if args.start_from_zero else None, None)
        self.y_axis_range = (0 if args.start_from_zero else None, args.y_axis_max_value)
        self.h_line = args.h_line
        self.h_line_label = args.h_line_label
        self.show_legend = args.show_legend
        self.legend_position = args.legend_position
        self.legend_font_size = args.legend_size
        self.output_name = args.output_file_path
        self.title = args.title
        self.show_subfigure_title = args.show_subfigure_title

    def apply_to_ax(self, ax: plt.Axes, metric: Metric) -> None:
        if self.h_line:
            ax.axhline(
                #y=self.h_line / 1000, color="r", linestyle="--", label=self.h_line_label
                y=self.h_line / 1000, color="r", linestyle="solid", label=self.h_line_label
            )

        ax.grid(which="major", linestyle="--", linewidth=0.1)

        y_axis_range = list(self.y_axis_range)
        if metric.y_max:
            y_axis_range[1] = metric.y_max
        ax.set_ylim(y_axis_range)


        if self.show_legend:
            loc = self.legend_position.replace("-", " ")
            ax.legend(
                scatterpoints=1,
                fontsize=self.legend_font_size,
                loc=loc,
            )

            if self.h_line:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(scatterpoints=1, fontsize=self.legend_font_size, loc=loc, handles=shift_list_to_right(handles), labels=shift_list_to_right(labels))


    def set_x_axis(self, ax, x_values: list[int], is_vbw: bool) -> None:
        x_values.sort()

        x_min = x_values[0]
        x_max = x_values[-1]

        if is_vbw:
            ax.set_xticks(range(x_min, x_max + 1, 8))
        else:
            ax.set_xticks(range(x_min, x_max + 1, 5))

        ax.set_xticks(range(x_min, x_max), minor=True)
        ax.tick_params(axis="x", which="minor", length=4, labelbottom=False)

        ax.set_xlim((x_min, x_max))

    def apply_to_figure(self) -> None:
        if self.title:
            plt.suptitle(self.title, fontsize=25)

    def output_graph(self) -> None:
        plt.tight_layout()
        if self.output_name is None:
            plt.show()
        elif self.output_name[-3:] == "eps":
            plt.savefig(self.output_name, format="eps")
        else:
            plt.savefig(self.output_name, format="png", bbox_inches="tight")


def read_file_format(text: str) -> tuple[str, pl.DataFrame]:
    runs_output, csv = text.split(OUTPUT_FILE_SECTION_BOUNDARY)
    runs_output = runs_output.split("\n")

    command = runs_output[0]

    for line in runs_output[1:]:
        if len(line.strip()) > 0:
            logging.critical("")
            logging.critical("")
            logging.critical(f"Reading failed run ({command}): {line}")
            logging.critical("")
            logging.critical("")

    data = pl.read_csv(StringIO(csv.strip()))
    return command, data


class ResultsFile:
    label: str
    command: str
    data: pl.DataFrame
    color: int

    def __init__(self, filename: str, label: str, color: int) -> None:
        self.label = label
        self.color = color
        with open(filename) as file:
            text = "".join([line for line in file])
            self.command, self.data = read_file_format(text)


def load_files(file_names_arg: str, labels_arg: str | None, colors: list[int]|None) -> list[ResultsFile]:
    file_names = file_names_arg.split(":")
    labels = labels_arg.split(":") if labels_arg is not None else file_names

    assert len(file_names) == len(labels) 

    return [
        ResultsFile(file_names[i], labels[i], i if colors is None else colors[i])
        for i in range(len(file_names))
    ]


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

    return "kernel_id", "Kernel ID"


def plot_scatter(
    datasets: list[ResultsFile], metrics: list[Metric], config: GraphConfiguration
):
    plt.style.use("classic")

    n_cols = len(metrics)
    n_cols_in_fig = math.ceil(n_cols**0.5)
    n_rows_in_fig = math.ceil(n_cols / n_cols_in_fig)

    figsize = (6 * n_cols_in_fig, 6 * n_rows_in_fig)
    fig, axes = plt.subplots(n_rows_in_fig, n_cols_in_fig, figsize=figsize)

    if n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    x_axis_column, pretty_x_axis_name = get_default_x_axis_column(datasets[0].data)

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for i, dataset in enumerate(datasets):
            x = dataset.data[x_axis_column]
            y = dataset.data[metric.name]

            y = list(map(metric.mutator, y))
            ax.scatter(
                x,
                y,
                s=26,
                linewidths=0,
                c=COLOR_SET[dataset.color],
                label=dataset.label,
            )

        ax.set_xlabel(pretty_x_axis_name)
        ax.set_ylabel(metric.column_alias)
        metric.formatter(ax)
        if config.show_subfigure_title:
            ax.set_title(metric.title_alias, fontsize=20)
        config.apply_to_ax(ax, metric)
        config.set_x_axis(ax, dataset.data[x_axis_column], x_axis_column == "vbw")

    for ax in axes[n_cols:]:
        ax.axis("off")

    config.apply_to_figure()

    config.output_graph()


def plot_bar(
    data: list[ResultsFile], metrics: list[Metric], config: GraphConfiguration
):
    pass

def get_metric(metric_arg: str, args: argparse.Namespace) -> Metric:
    metric = NVPROF_METRICS_MAPPING[metric_arg]

    if args.throughput:
        default_mutator = metric.mutator
        metric.mutator = lambda x: args.throughput / default_mutator(x)

        if args.throughput_label:
            metric.column_alias = args.throughput_label

    return metric

PLOT_FUNCTION_MAPPING = {
    GraphTypes.SCATTER: plot_scatter,
    GraphTypes.BAR: plot_bar,
}


def main(args):
    colors = None
    if args.colors is not None:
        colors = list(map(int,args.colors.split(':')))

    results = load_files(args.files, args.labels, colors)
    assert len(results) > 0

    columns = parse_column_names(args.columns, results)
    assert len(columns) > 0

    

    metrics = [get_metric(column, args) for column in columns]

    config = GraphConfiguration(args)

    PLOT_FUNCTION_MAPPING[args.plot](results, metrics, config)


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
        "-tp",
        "--throughput",
        type=int,
        help=f"Specify to plot the data as throughput in unit of column. The number is the amount of work units.",
    )
    parser.add_argument(
        "-tpl",
        "--throughput-label",
        type=str,
        help=f"Specify the label to be used on the throughput axis",
    )
    parser.add_argument(
        "-o",
        "--output-file-path",
        type=str,
        default=None,
        help=f"Filename to save graph to",
    )
    parser.add_argument(
        "-l", "--labels", type=str, help="Labels corresponding to each file. Delimited by ':'"
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
        "-ls",
        "--legend-size",
        type=int,
        default=10,
        help="Defines the fontsize of the legend",
    )
    parser.add_argument(
        "-c",
        "--colors",
        type=str,
        help="Defines the colors used, must have equal length to number of files",
    )
    parser.add_argument(
        "-sst",
        "--show-subfigure-title",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Show subfigure title",
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
        "-hll",
        "--h-line-label",
        type=str,
        default="Baseline",
        help="Horizontal, dashed line, label name",
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
