#!/usr/bin/env python3

import os
import sys

import argparse
import logging
import itertools

import inspect
import types
from typing import Any
import subprocess
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math

FLOAT_BYTES = 4
GB_BYTES = 1024 * 1024 * 1024
MS_IN_S = 1000

VECTOR_SIZE = 1024

COLOR_SET = [
    "tab:blue",
    "#66bf5c",  # Light green
    "#177314",  # Darker green
    "tab:orange",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "tab:green",
]


def directory_exists(path: str) -> bool:
    return Path(path).is_dir()


def get_plotting_functions() -> list[tuple[str, types.FunctionType]]:
    current_module = inspect.getmodule(inspect.currentframe())
    functions = inspect.getmembers(current_module, inspect.isfunction)

    plotting_function_prefix = "plot_"
    plotting_functions = filter(
        lambda x: x[0].startswith(plotting_function_prefix), functions
    )
    stripped_prefixes_from_name = map(
        lambda x: (x[0].replace(plotting_function_prefix, ""), x[1]), plotting_functions
    )
    return list(stripped_prefixes_from_name)


@dataclass
class DataSource:
    label: str
    color: int
    x_data: list[Any]
    y_data: list[Any]


@dataclass
class GroupedDataSource:
    g: tuple
    color: int
    x_data: list[Any]
    y_data: list[Any]
    label: str | None = None

    def set_label(self, label: str):
        self.label = label
        return self


def create_scatter_graph(
    data_sources: list[DataSource],
    x_label: str,
    y_label: str,
    out: str,
    y_lim: tuple[int, int] | None = None,
    octal_grid: bool = False,
    figsize: tuple[int, int] = (5, 5),
    legend_pos: str = "best",
):
    fig, ax = plt.subplots(figsize=figsize)
    x_min = min(data_sources[0].x_data)
    x_max = max(data_sources[0].x_data)
    for source in data_sources:
        x_min = min(source.x_data)
        x_max = max(source.x_data)
        ax.scatter(
            source.x_data,
            source.y_data,
            s=26,
            linewidths=0,
            c=COLOR_SET[source.color],
            label=source.label,
        )

    ax.set_xticks(range(math.floor(x_min), math.ceil(x_max)), minor=True)
    ax.tick_params(axis="x", which="minor", length=4, labelbottom=False)
    if octal_grid:
        ax.set_xticks(range(math.floor(x_min), math.ceil(x_max), 8))

    ax.grid(which="major", linestyle="--", linewidth=0.1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)

    ax.legend(loc=legend_pos)

    fig.savefig(out, dpi=300, format="eps", bbox_inches="tight")
    plt.close(fig)


def plot_ffor(input_dir: str, output_dir: str):
    df = pl.read_csv(os.path.join(input_dir, "ffor.csv"))
    df = df.with_columns(
        ((pl.col("n_vecs") * VECTOR_SIZE) / pl.col("duration (ns)")).alias(
            "throughput"
        ),
        (pl.col("duration (ns)") / 1000).alias("duration (us)"),
    )

    sources = [
        GroupedDataSource(
            label_data[0],
            0,
            label_data[1].get_column("vbw").to_list(),
            label_data[1].get_column("duration (us)").to_list(),
        )
        for label_data in df.group_by(
            ["data_type", "kernel", "unpack_n_vectors", "unpacker"],
        )
    ]

    graph_groups = {
        "stateless": list(
            map(
                lambda x: x.set_label(f"{x.g[3]}"),
                filter(
                    lambda x: x.g[0] == "u32"
                    and x.g[1] == "query"
                    and x.g[2] == 1
                    and x.g[3] == "stateless",
                    sources,
                ),
            )
        ),
        "stateful_branchless_u32-vs-u64": list(
            map(
                lambda x: x.set_label(f"{x.g[0]}"),
                filter(
                    lambda x: x.g[1] == "query"
                    and x.g[2] == 1
                    and x.g[3] == "stateful_branchless",
                    sources,
                ),
            )
        ),
    }

    for name, graph_sources in graph_groups.items():
        y_lim = max([max(source.y_data) for source in graph_sources]) * 1.1

        for i, source in enumerate(graph_sources):
            source.color = i

        create_scatter_graph(
            graph_sources,
            "Value bit width",
            "Duration (us)",
            os.path.join(output_dir, f"ffor-{name}.eps"),
            y_lim=(0, y_lim),
            legend_pos="lower left",
        )


def plot_alp_ec(input_dir: str, output_dir: str):
    df = pl.read_csv(os.path.join(input_dir, "alp-ec.csv"))
    df = df.with_columns(
        ((pl.col("n_vecs") * VECTOR_SIZE) / pl.col("duration (ns)")).alias(
            "throughput"
        ),
        (pl.col("duration (ns)") / 1000).alias("duration (us)"),
    )

    sources = [
        GroupedDataSource(
            label_data[0],
            0,
            label_data[1].get_column("ec").to_list(),
            label_data[1].get_column("duration (us)").to_list(),
        )
        for label_data in df.group_by(
            ["data_type", "kernel", "unpack_n_vectors", "unpacker", "patcher"],
        )
    ]

    graph_groups = {
        "stateless": list(
            map(
                lambda x: x.set_label(f"{x.g[3]}"),
                filter(
                    lambda x: x.g[0] == "f32"
                    and x.g[1] == "query"
                    and x.g[2] == 1
                    and x.g[3] == "stateless"
                    and x.g[4] == "stateless",
                    sources,
                ),
            )
        ),
        "stateful_branchless_prefetch_all_f32-vs-f64": list(
            map(
                lambda x: x.set_label(f"{x.g[0]}"),
                filter(
                    lambda x: x.g[1] == "query"
                    and x.g[2] == 1
                    and x.g[3] == "stateful_branchless"
                    and x.g[4] == "prefetch_all",
                    sources,
                ),
            )
        ),
    }

    for name, graph_sources in graph_groups.items():
        y_lim = max([max(source.y_data) for source in graph_sources]) * 1.1

        for i, source in enumerate(graph_sources):
            source.color = i

        create_scatter_graph(
            graph_sources,
            "Exception count",
            "Duration (us)",
            os.path.join(output_dir, f"alp-ec-{name}.eps"),
            y_lim=(0, y_lim),
            legend_pos="lower left",
        )


def main(args):
    assert directory_exists(args.input_dir)
    assert directory_exists(args.output_dir)

    _ = args.plotting_function(args.input_dir, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    plotting_functions = {func[0]: func[1] for func in get_plotting_functions()}
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "plotting_function",
        type=str,
        choices=list(plotting_functions.keys()) + ["all"],
        help="function to execute",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="directory_to_write_results_to",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="directory_to_write_results_to",
    )
    parser.add_argument(
        "-dr",
        "--dry-run",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Dry run",
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

    if args.plotting_function == "all":
        args.plotting_function = lambda in_dir, out_dir: list(
            func(in_dir, out_dir) for func in plotting_functions.values()
        )
    else:
        args.plotting_function = plotting_functions[args.plotting_function]
    main(args)
