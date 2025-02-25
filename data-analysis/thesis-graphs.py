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


"""
FLS_QUERY_FILES = [
    "fls_query-noninterleaved-1-1",
    "fls_query-noninterleaved-4-1",
    "fls_query-stateful_branchless-1-1",
    "fls_query-stateful_branchless-4-1",
    "fls_query-stateful-cache-1-1",
    "fls_query-stateful-cache-4-1",
    "fls_query-stateful-local-1-1-1",
    "fls_query-stateful-local-1-4-1",
    "fls_query-stateful-local-2-1-1",
    "fls_query-stateful-local-2-4-1",
    "fls_query-stateful-local-4-1-1",
    "fls_query-stateful-local-4-4-1",
    "fls_query-stateful-register-1-1-1",
    "fls_query-stateful-register-1-4-1",
    "fls_query-stateful-register-2-1-1",
    "fls_query-stateful-register-2-4-1",
    "fls_query-stateful-register-4-1-1",
    "fls_query-stateful-register-4-4-1",
    "fls_query-stateful-register-branchless-1-1-1",
    "fls_query-stateful-register-branchless-1-4-1",
    "fls_query-stateful-register-branchless-2-1-1",
    "fls_query-stateful-register-branchless-2-4-1",
    "fls_query-stateful-register-branchless-4-1-1",
    "fls_query-stateful-register-branchless-4-4-1",
    "fls_query-stateless-1-1",
    "fls_query-stateless-4-1",
    "fls_query-stateless_branchless-1-1",
    "fls_query-stateless_branchless-4-1",
]

ALP_QUERY_FILES = [
"alp_query-stateless-prefetch_position-4-1",
]
"""


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
                "fls_query-stateful-cache-1-1",
                "fls_query-stateful-local-1-1-1",
                "fls_query-stateful-register-1-1-1",
                "fls_query-stateful-register-branchless-1-1-1",
            ],
            [
                "cache-0b-1v",
                "local-1b-1v",
                "register-1b-1v",
                "register-branchless-1b-1v",
            ],
            "stateful-v1-b1",
        ),
        GraphDefinition(
            [
                "fls_query-stateful-local-2-1-1",
                "fls_query-stateful-register-2-1-1",
                "fls_query-stateful-register-branchless-2-1-1",
            ],
            ["local-2b-1v", "register-2-1v", "register-branchless-2b-1v"],
            "stateful-v1-b2",
            colors=list(range(1, 4)),
        ),
        GraphDefinition(
            [
                "fls_query-stateful-local-4-1-1",
                "fls_query-stateful-register-4-1-1",
                "fls_query-stateful-register-branchless-4-1-1",
            ],
            ["local-4b-1v", "register-4b-1v", "register-branchless-4b-1v"],
            "stateful-v1-b4",
            colors=list(range(1, 4)),
        ),
        GraphDefinition(
            [
                "fls_query-stateful-cache-4-1",
                "fls_query-stateful-local-1-4-1",
                "fls_query-stateful-register-1-4-1",
                "fls_query-stateful-register-branchless-1-4-1",
            ],
            [
                "cache-0b-4v",
                "local-1b-4v",
                "register-1b-4v",
                "register-branchless-1b-4v",
            ],
            "stateful-v4-b1",
        ),
        GraphDefinition(
            [
                "fls_query-stateful-local-2-4-1",
                "fls_query-stateful-register-2-4-1",
                "fls_query-stateful-register-branchless-2-4-1",
            ],
            ["local-2b-4v", "register-2b-4v", "register-branchless-2b-4v"],
            "stateful-v4-b2",
            colors=list(range(1, 4)),
        ),
        GraphDefinition(
            [
                "fls_query-stateful-local-4-4-1",
                "fls_query-stateful-register-4-4-1",
                "fls_query-stateful-register-branchless-4-4-1",
            ],
            ["local-4b-4v", "register-4b-4v", "register-branchless-4b-4v"],
            "stateful-v4-b4",
            colors=list(range(1, 4)),
        ),
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


def plot_all_unpackers(input_dir: str, output_dir: str):
    graphs = [
        GraphDefinition(
            [
                "fls_query-stateless-1-1",
                "fls_query-stateful-register-branchless-2-1-1",
                "fls_query-stateless_branchless-1-1",
                "fls_query-stateful_branchless-1-1",
            ],
            [
                "stateless-1v",
                "stateful-1v",
                "stateless-branchless-1v",
                "stateful-branchless-1v",
            ],
            "unpackers-1v",
        ),
        GraphDefinition(
            [
                "fls_query-stateless-4-1",
                "fls_query-stateful-register-branchless-2-4-1",
                "fls_query-stateless_branchless-4-1",
                "fls_query-stateful_branchless-4-1",
            ],
            [
                "stateless-4v",
                "stateful-4v",
                "stateless-branchless-4v",
                "stateful-branchless-4v",
            ],
            "unpackers-4v",
        ),
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


def plot_all_patchers(input_dir: str, output_dir: str):
    def get_filename(unpacker: str, patcher: str, n_vecs: int) -> str:
        return f"alp_query-{unpacker}-{patcher}-{n_vecs}-1"

    counter = 0
    def get_graphs_for_patchers(patchers: list[str], out: str, color_offset: int) -> list[GraphDefinition]:
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
                colors=range(color_offset, color_offset + len(patchers))
            )
            for n_vecs in [1, 4]
        ]

    nonparallel_patchers = [
        "stateless",
        "stateful",
    ], "nonparallel-exception-patchers", 2500, 0
    parallel_patchers = [
        "naive",
        "naive_branchless",
    ], "parallel-exception-patchers", 400, 0
    prefetch_parallel_patchers = [
        "prefetch_position",
        "prefetch_all",
        "prefetch_all_branchless",
    ], "prefetch-parallel-exception-patchers", 400, len(parallel_patchers[0])

    for patcher_set, out, yamv, color_offset in [nonparallel_patchers, parallel_patchers, prefetch_parallel_patchers]:
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
