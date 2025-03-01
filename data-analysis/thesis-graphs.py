#!/usr/bin/python3

import os
import sys

import argparse
import logging
import itertools

import inspect
import types
import subprocess
from pathlib import Path
from typing import Iterable

GRAPHER_PATH = "./data-analysis/graph-generator.py"


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


def execute_command(command: str) -> str:
    if args.dry_run:
        print(command, file=sys.stderr)
        return ""

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    logging.info(f"EXECUTED: {command}")

    if result.returncode != 0:
        logging.critical(f"Exited with code {result.returncode}: {command}")
        logging.critical(f"STDOUT: {result.stdout}")
        logging.critical(f"STDERR: {result.stderr}")
        exit(0)

    return result.stdout


class GraphDefinition:
    files: list[str]
    labels: list[str]
    out: str
    title: str | None
    colors: Iterable[int]

    def __init__(
        self,
        files: list[str],
        labels: list[str],
        out: str,
        title: str | None = None,
        colors: Iterable[int] | None = None,
    ) -> None:
        self.files = files
        self.labels = labels
        self.out = out
        self.title = title
        self.colors = colors if colors else range(len(self.files))


def plot_stateful_unpackers(input_dir: str, output_dir: str):
    graphs = [
        GraphDefinition(
            [
                f"fls_query-stateful-cache-{n_vecs}-1",
                f"fls_query-stateful-local-1-{n_vecs}-1",
                f"fls_query-stateful-register-1-{n_vecs}-1",
                f"fls_query-stateful-register-branchless-1-{n_vecs}-1",
            ],
            [
                f"cache-0b-{n_vecs}v",
                f"local-1b-{n_vecs}v",
                f"register-1b-{n_vecs}v",
                f"register-branchless-1b-{n_vecs}v",
            ],
            f"stateful-v{n_vecs}-b1",
        )
        for n_vecs in [1, 4]
    ] + [
        GraphDefinition(
            [
                f"fls_query-stateful-local-{buffer_size}-{n_vecs}-1",
                f"fls_query-stateful-register-{buffer_size}-{n_vecs}-1",
                f"fls_query-stateful-register-branchless-{buffer_size}-{n_vecs}-1",
            ],
            [
                f"local-{buffer_size}b-{n_vecs}v",
                f"register-{buffer_size}b-{n_vecs}v",
                f"register-branchless-{buffer_size}b-{n_vecs}v",
            ],
            f"stateful-v{n_vecs}-b{buffer_size}",
            colors=list(range(1, 4)),
        )
        for buffer_size in [2, 4]
        for n_vecs in [1, 4]
    ]

    for graph in graphs:
        execute_command(
            f'{GRAPHER_PATH} {":".join([os.path.join(input_dir, file) for file in graph.files])} '
            f'scatter execution_time -l {":".join(graph.labels)} '
            f'-hl 250488 -hll "Normal execution" '
            f"-lp upper-left -yamv 700 "
            f'-c {":".join(map(str, graph.colors))} '
            f'-o {os.path.join(output_dir,f"{graph.out}.eps")}'
        )


def inner_plot_all_unpackers(input_dir: str, output_dir: str, experiment: str):
    graphs = [
        GraphDefinition(
            [
                f"{experiment}-stateless-{n_vecs}-1",
                f"{experiment}-stateful-register-branchless-2-{n_vecs}-1",
                f"{experiment}-stateless_branchless-{n_vecs}-1",
                f"{experiment}-stateful_branchless-{n_vecs}-1",
            ],
            [
                f"stateless-{n_vecs}v",
                f"stateful-{n_vecs}v",
                f"stateless-branchless-{n_vecs}v",
                f"stateful-branchless-{n_vecs}v",
            ],
            f"unpackers-{experiment}-{n_vecs}v",
        )
        for n_vecs in [1, 4]
    ]

    for graph in graphs:
        execute_command(
            f'{GRAPHER_PATH} {":".join([os.path.join(input_dir, file) for file in graph.files])} '
            f'scatter execution_time -l {":".join(graph.labels)} '
            f'-hl 250488 -hll "Normal execution" '
            f"-lp upper-left -yamv 300 "
            f'-c {":".join(map(str, graph.colors))} '
            f'-o {os.path.join(output_dir,f"{graph.out}.eps")}'
        )

def plot_all_unpackers_query(input_dir: str, output_dir: str):
    inner_plot_all_unpackers(input_dir, output_dir, "fls_query")


def plot_all_unpackers_decompress(input_dir: str, output_dir: str):
    inner_plot_all_unpackers(input_dir, output_dir, "fls_decompress")


def plot_all_unpackers_compute(input_dir: str, output_dir: str):
    inner_plot_all_unpackers(input_dir, output_dir, "fls_compute")


def plot_all_patchers(input_dir: str, output_dir: str):
    def get_filename(unpacker: str, patcher: str, n_vecs: int) -> str:
        return f"alp_query-{unpacker}-{patcher}-{n_vecs}-1"

    def get_graphs_for_patchers(
        patchers: list[str], out: str, color_offset: int
    ) -> list[GraphDefinition]:
        return [
            GraphDefinition(
                [
                    get_filename(
                        (
                            "stateful_branchless"
                            if n_vecs == 1
                            else "stateful-register-2"
                        ),
                        patcher,
                        n_vecs,
                    )
                    for patcher in patchers
                ],
                patchers,
                out=out + f"-v{n_vecs}",
                colors=range(color_offset, color_offset + len(patchers)),
            )
            for n_vecs in [1, 4]
        ]

    nonparallel_patchers = (
        [
            "stateless",
            "stateful",
        ],
        "nonparallel-exception-patchers",
        2500,
        0,
    )
    parallel_patchers = (
        [
            "naive",
            "naive_branchless",
        ],
        "parallel-exception-patchers",
        400,
        0,
    )
    prefetch_parallel_patchers = (
        [
            "prefetch_position",
            "prefetch_all",
            "prefetch_all_branchless",
        ],
        "prefetch-parallel-exception-patchers",
        400,
        len(parallel_patchers[0]),
    )

    for patcher_set, out, yamv, color_offset in [
        nonparallel_patchers,
        parallel_patchers,
        prefetch_parallel_patchers,
    ]:
        for graph in get_graphs_for_patchers(patcher_set, out, color_offset):
            execute_command(
                f'{GRAPHER_PATH} {":".join([os.path.join(input_dir, file) for file in graph.files])} '
                f'scatter execution_time -l {":".join(graph.labels)} '
                f'-hl 250488 -hll "Normal execution" '
                f"-lp lower-right -yamv {yamv} "
                f'-c {":".join(map(str, graph.colors))} '
                f'-o {os.path.join(output_dir,f"{graph.out}.eps")}'
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
        choices=list(plotting_functions.keys()),
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
    args.plotting_function = plotting_functions[args.plotting_function]
    main(args)
