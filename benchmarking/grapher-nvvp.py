#!/usr/bin/env python3

from collections.abc import Callable
import os
import sys

import itertools
import argparse
import logging
import copy

import inspect
import types
from typing import Any, Iterable, Optional
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math

FLOAT_BYTES = 4
GB_BYTES = 1024 * 1024 * 1024
MS_IN_S = 1000

VECTOR_SIZE = 1024

COLOR_SET = [
    "tab:blue",
    "tab:orange",
    "tab:red",
    "tab:green",
    "tab:pink",
    "tab:purple",
    "tab:brown",
    "tab:olive",
    "tab:gray",
    "tab:cyan",
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
    group_by_column_values: tuple
    x_data: list[Any]
    y_data: list[Any]
    color: int = 0
    label: str | None = None

    def set_label(self, label: str):
        self.label = label
        return self


def create_scatter_graph(
    data_sources: list[DataSource | GroupedDataSource],
    x_label: str,
    y_label: str,
    out: str,
    y_lim: tuple[int | float, int | float] | None = None,
    octal_grid: bool = False,
    figsize: tuple[int, int] = (5, 5),
    legend_pos: str = "best",
    add_lines: bool = False,
    x_axis_percentage: bool = False,
    title: Optional[str] = None,
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
        if add_lines:
            ax.plot(
                source.x_data,
                source.y_data,
                c=COLOR_SET[source.color],
            )

    if x_axis_percentage:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xticks(range(math.floor(x_min), math.ceil(x_max)), minor=True)
    ax.tick_params(axis="x", which="minor", length=4, labelbottom=False)
    if octal_grid:
        ax.set_xticks(range(math.floor(x_min), math.ceil(x_max) + 1, 8))

    ax.grid(which="major", linestyle="--", linewidth=0.1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)

    if title:
        ax.set_title(title)

    ax.legend(loc=legend_pos)

    fig.savefig(out, dpi=300, format="eps", bbox_inches="tight")
    plt.close(fig)


def create_multi_bar_graph(
    data_sources: list[DataSource | GroupedDataSource],
    bargroup_labels: list[str] | None,
    x_label: str,
    y_label: str,
    out: str,
    y_lim: tuple[int, int] | None = None,
    colors: list[int] | None = None,
):
    n_bars = len(data_sources)
    n_groups = len(data_sources[0].x_data)
    assert all(len(x.x_data) == n_groups for x in data_sources)

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.8 / n_bars
    indices = np.arange(n_groups)
    for i, source in enumerate(data_sources):
        bar_label = source.label
        bar_data = source.y_data

        positions = indices + (i - n_bars / 2 + 0.5) * bar_width
        ax.bar(
            positions,
            bar_data,
            bar_width,
            label=bar_label,
            color=COLOR_SET[colors[i]] if colors else COLOR_SET[i],
        )

    if bargroup_labels:
        ax.set_xticks(indices)
        ax.set_xticklabels(bargroup_labels)
    else:
        plt.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)
    ax.legend()

    fig.savefig(out, dpi=300, format="eps", bbox_inches="tight")
    plt.close(fig)


def reorder_columns(df: pl.DataFrame, reference_columns: list[str]) -> pl.DataFrame:
    ordered_columns = [col for col in reference_columns if col in df.columns]
    return df.select(ordered_columns)


def average_samples(df: pl.DataFrame, columns_to_avg: list[str]) -> pl.DataFrame:
    dropped_columns = ["kernel_index", "sample_run"]
    reference_columns = [col for col in df.columns if col not in dropped_columns]
    group_by_cols = [
        col
        for col in df.columns
        if col not in columns_to_avg and col not in dropped_columns
    ]

    df = df.group_by(group_by_cols, maintain_order=True).agg(
        [pl.col(col).mean().alias(col) for col in columns_to_avg]
    )

    df = reorder_columns(df, reference_columns)
    return df


def create_grouped_data_sources(
    df: pl.DataFrame, group_by_columns: list[str], x_column: str, y_column: str
) -> list[GroupedDataSource]:
    return [
        GroupedDataSource(
            group_by_column_values,
            data.get_column(x_column).to_list(),
            data.get_column(y_column).to_list(),
        )
        for group_by_column_values, data in df.group_by(
            group_by_columns, maintain_order=True
        )
    ]


def define_graph(
    sources: list[GroupedDataSource],
    filters: list[list[Any] | Any],
    label_func: Callable[[tuple], str],
) -> list[GroupedDataSource]:
    assert len(sources[0].group_by_column_values) == len(
        filters
    ), "Mismatch filter length and group by length"
    filter_lists = list(map(lambda x: x if isinstance(x, list) else [x], filters))

    filtered = filter(
        lambda source: all(
            [
                source.group_by_column_values[i] in filter or len(filter) == 0
                for i, filter in enumerate(filter_lists)
            ]
        ),
        sources,
    )

    labelled = list(
        map(
            lambda x: copy.copy(x.set_label(label_func(x.group_by_column_values))),
            filtered,
        )
    )

    return sorted(labelled, key=lambda x: x.label)


@dataclass
class SourceSet:
    file_name: str
    sources: list[GroupedDataSource]
    title: Optional[str] = None
    colors: Optional[Iterable[int]] = None


def calculate_common_y_lim(sources: Iterable[DataSource | GroupedDataSource]) -> float:
    all_y_data = itertools.chain.from_iterable(map(lambda x: x.y_data, sources))
    return max(all_y_data) * 1.1


def assign_colors(
    sources: Iterable[DataSource | GroupedDataSource],
    colors: Optional[Iterable[int]] = None,
) -> Iterable[DataSource | GroupedDataSource]:
    if colors:
        for c, source in zip(colors, sources):
            source.color = c
    else:
        for c, source in enumerate(sources):
            source.color = c

    return sources


def convert_str_to_label(label: str) -> str:
    return label.replace("_", " ").title()


def plot_ffor(input_dir: str, output_dir: str):
    df = pl.read_csv(os.path.join(input_dir, "ffor.csv"))
    df = average_samples(df, ["duration_ns"])
    df = df.with_columns(
        (pl.col("duration_ns") / 1000).alias("duration_us"),
        (pl.col("n_vecs") / (pl.col("duration_ns") / 1000)).alias("throughput"),
    )

    sources = create_grouped_data_sources(
        df,
        ["data_type", "kernel", "unpack_n_vectors", "unpacker"],
        "vbw",
        "throughput",
    )

    stateful_storage_types = [
        "cache",
        "local",
        "shared",
        "register",
        "register_branchless",
    ]
    source_sets = (
        [
            SourceSet(
                f"32-switch-vs-1-switch-v1-u32",
                define_graph(
                    sources,
                    [
                        "u32",
                        "query",
                        1,
                        [
                            "old_fls",
                            "switch_case",
                        ],
                    ],
                    lambda x: (
                        "FastLanesOnGPU"
                        if x[3] == "old_fls"
                        else "One value decoding switch"
                    ),
                ),
                title=f"u32, Concurrent Vectors: 1",
                colors=range(0, 2),
            )
        ]
        + [
            SourceSet(
                f"old-fls-vs-stateless-{kernel}-v{n_vec}-{data_type}",
                define_graph(
                    sources,
                    [
                        data_type,
                        kernel,
                        n_vec,
                        [
                            "old_fls" if n_vec == 1 and data_type == "u32" else "",
                            "switch_case" if n_vec == 1 and data_type == "u32" else "",
                            "stateless",
                        ],
                    ],
                    lambda x: (
                        "FastLanesOnGPU"
                        if x[3] == "old_fls"
                        else (
                            "One value decoding switch"
                            if x[3] == "switch_case"
                            else "Stateless"
                        )
                    ),
                ),
                title=f"{data_type}, Concurrent Vectors: {n_vec}",
                colors=range(0 if n_vec == 1 and data_type == "u32" else 2, 3),
            )
            for kernel, n_vec, data_type in itertools.product(
                ["query", "decompress"], [1, 4], ["u32"]
            )
        ]
        + [
            SourceSet(
                f"stateful-b{buffer_size}-v{n_vec}-{data_type}",
                define_graph(
                    sources,
                    [
                        data_type,
                        "query",
                        n_vec,
                        [
                            (
                                (f"stateful_{storage_type}" if buffer_size == 1 else "")
                                if storage_type == "cache"
                                else f"stateful_{storage_type}_{buffer_size}"
                            )
                            for storage_type in stateful_storage_types
                        ],
                    ],
                    lambda x: convert_str_to_label(" ".join(x[3].split("_")[1:-1])),
                ),
                title=f"{data_type}, Concurrent Vectors: {n_vec}, Buffer Size: {buffer_size}",
                colors=range(0 if buffer_size == 1 else 1, len(stateful_storage_types)),
            )
            for buffer_size, n_vec, data_type in itertools.product(
                [1, 2, 4],
                [1, 4],
                ["u32", "u64"],
            )
        ]
        + [
            SourceSet(
                f"all-{kernel}-v{n_vec}-{data_type}",
                define_graph(
                    sources,
                    [
                        data_type,
                        kernel,
                        n_vec,
                        [
                            "old_fls" if n_vec == 1 and data_type == "u32" else "",
                            "stateless",
                            "stateless_branchless",
                            "stateful_register_branchless_2",
                            "stateful_branchless",
                        ],
                    ],
                    lambda x: (
                        "FastLanesOnGPU"
                        if x[3] == "old_fls"
                        else (
                            convert_str_to_label(
                                convert_str_to_label(x[3])
                                if x[3] != "stateful_register_branchless_2"
                                else "Stateful"
                            )
                        )
                    ),
                ),
                title=f"{data_type}, Concurrent Vectors: {n_vec}",
                colors=range(0 if n_vec == 1 and data_type == "u32" else 1, 5),
            )
            for kernel, n_vec, data_type in itertools.product(
                ["query", "decompress"], [1, 4], ["u32", "u64"]
            )
        ]
    )

    y_lim = calculate_common_y_lim(
        itertools.chain.from_iterable(map(lambda x: x.sources, source_sets))
    )

    for source_set in source_sets:
        sources = assign_colors(source_set.sources, source_set.colors)

        create_scatter_graph(
            sources,
            "Value bit width",
            "Throughput (vecs/us)",
            os.path.join(output_dir, f"ffor-{source_set.file_name}.eps"),
            y_lim=(0, y_lim),
            octal_grid=True,
            title=source_set.title,
        )


def plot_alp_ec(input_dir: str, output_dir: str):
    df = pl.read_csv(os.path.join(input_dir, "alp-ec.csv"))
    df = average_samples(df, ["duration_ns"])
    df = df.with_columns(
        (pl.col("duration_ns") / 1000).alias("duration_us"),
        (pl.col("n_vecs") / (pl.col("duration_ns") / 1000)).alias("throughput"),
    )

    sources = create_grouped_data_sources(
        df,
        ["data_type", "kernel", "unpack_n_vectors", "unpacker", "patcher"],
        "ec",
        "throughput",
    )

    alp_unpacker = "stateful_branchless"
    source_sets = (
        [
            SourceSet(
                f"alp-{kernel}-v{n_vec}-{data_type}",
                define_graph(
                    sources,
                    [
                        data_type,
                        kernel,
                        n_vec,
                        alp_unpacker,
                        [
                            "stateless",
                            "stateful",
                        ],
                    ],
                    lambda x: convert_str_to_label(x[4]),
                ),
                title=f"{data_type}, Concurrent Vectors: {n_vec}",
                colors=range(0, 2),
            )
            for kernel, n_vec, data_type in itertools.product(
                ["query"], [1, 4], ["f32", "f64"]
            )
        ] + [
            SourceSet(
                f"galp-{kernel}-v{n_vec}-{data_type}",
                define_graph(
                    sources,
                    [
                        data_type,
                        kernel,
                        n_vec,
                        alp_unpacker,
                        [
                            "naive",
                            "naive_branchless",
                            "prefetch_position",
                            "prefetch_all",
                            "prefetch_all_branchless",
                        ],
                    ],
                    lambda x: convert_str_to_label(x[4]),
                ),
                title=f"{data_type}, Concurrent Vectors: {n_vec}",
                colors=range(0, 5),
            )
            for kernel, n_vec, data_type in itertools.product(
                ["query"], [1, 4], ["f32", "f64"]
            )
        ] 
    )

    y_lim = calculate_common_y_lim(
        itertools.chain.from_iterable(map(lambda x: x.sources, source_sets))
    )


    for source_set in source_sets:
        sources = assign_colors(source_set.sources, source_set.colors)

        create_scatter_graph(
            sources,
            "Exception count",
            "Throughput (vecs/us)",
            os.path.join(output_dir, f"alp-ec-{source_set.file_name}.eps"),
            y_lim=(0, y_lim),
            title=source_set.title,
        )


def plot_multi_column(input_dir: str, output_dir: str):
    df = pl.read_csv(os.path.join(input_dir, "multi-column.csv"))
    df = average_samples(df, ["duration_ns"])
    df = df.with_columns(
        (
            (pl.col("n_vecs") * VECTOR_SIZE * pl.col("n_cols")) / pl.col("duration_ns")
        ).alias("throughput"),
    )

    sources = create_grouped_data_sources(
        df,
        ["data_type", "unpack_n_vectors", "unpacker", "patcher"],
        "n_cols",
        "throughput",
    )

    graphs = {
        "ffor": define_graph(
            sources,
            [
                "u32",
                [1, 4],
                "stateful_branchless",
                "none",
            ],
            lambda x: f"{x[2]}-{x[1]}-vecs",
        ),
        "alp": define_graph(
            sources,
            [
                "f32",
                [1, 4],
                "stateful_branchless",
                ["prefetch_all", "prefetch_all_branchless"],
            ],
            lambda x: f"{x[3]}-{x[1]}-vecs",
        ),
    }

    for name, graph_sources in graphs.items():
        graph_sources = assign_colors(graph_sources)

        create_multi_bar_graph(
            graph_sources,
            list(map(str, range(1, 10 + 1))),
            "Number of columns",
            "Throughput (vectors/ns/column)",
            os.path.join(output_dir, f"multi-column-throughput-{name}.eps"),
            y_lim=(0, calculate_common_y_lim(graph_sources)),
        )


def plot_compressors(input_dir: str, output_dir: str):
    df = pl.read_csv(os.path.join(input_dir, "compressors.csv"))
    df = average_samples(
        df,
        [
            "duration_ms",
            "avg_bits_per_value",
            "avg_exceptions_per_vector",
            "compression_ratio",
        ],
    )
    df = df.with_columns(
        (pl.col("n_bytes") / pl.col("duration_ms")).alias("throughput")
    )

    sources = create_grouped_data_sources(
        df,
        ["kernel", "compressor", "data_type"],
        "compression_ratio",
        "throughput",
    )

    graphs = {
        "scatter-compression-vs-throughput-f32": define_graph(
            sources,
            [
                "decompression_query",
                [],
                "f32",
            ],
            lambda x: f"{x[1]}",
        ),
        "scatter-compression-vs-throughput-f64": define_graph(
            sources,
            [
                "decompression_query",
                [],
                "f64",
            ],
            lambda x: f"{x[1]}",
        ),
    }

    for name, graph_sources in graphs.items():
        graph_sources = assign_colors(graph_sources)

        create_scatter_graph(
            graph_sources,
            "Compression ratio",
            "Throughput",
            os.path.join(output_dir, f"compressors-{name}.eps"),
            y_lim=(0, calculate_common_y_lim(graph_sources)),
            legend_pos="upper right",
        )


def plot_ilp_experiment(input_dir: str, output_dir: str):
    MAX_THREAD_BLOCK_SIZE = 1024
    TOTAL_PTRS_CHASED = 100 * 10 * 1024 * 1024
    df = pl.read_csv(os.path.join(input_dir, "ilp-experiment.csv"))
    df = average_samples(df, ["duration_ns"])
    df = df.with_columns(
        (TOTAL_PTRS_CHASED / pl.col("duration_ns")).alias("throughput"),
        (pl.col("duration_ns") / 1000).alias("duration_us"),
        (pl.col("threads/block") / MAX_THREAD_BLOCK_SIZE).alias("occupancy"),
    )

    sources = [
        DataSource(
            f"{label_data[0][0]} concurrent ptrs/warp",
            i,
            label_data[1].get_column("occupancy").to_list(),
            label_data[1].get_column("duration_us").to_list(),
        )
        for i, label_data in enumerate(df.group_by(["ilp"], maintain_order=True))
    ]

    create_scatter_graph(
        sources,
        "Occupancy (%)",
        "Duration (us)",
        os.path.join(output_dir, "ilp-experiment-duration.eps"),
        legend_pos="upper right",
        add_lines=True,
        x_axis_percentage=True,
    )

    sources = [
        DataSource(
            f"{label_data[0][0]} concurrent ptrs/warp",
            i,
            label_data[1].get_column("occupancy").to_list(),
            label_data[1].get_column("throughput").to_list(),
        )
        for i, label_data in enumerate(df.group_by(["ilp"], maintain_order=True))
    ]
    create_scatter_graph(
        sources,
        "Occupancy (%)",
        "Throughput (ptrs/ns)",
        os.path.join(output_dir, "ilp-experiment-throughput.eps"),
        legend_pos="lower right",
        add_lines=True,
        x_axis_percentage=True,
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
