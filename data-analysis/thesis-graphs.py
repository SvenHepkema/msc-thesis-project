#!/usr/bin/python3

import os
import sys

import argparse
import logging

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
ALL_FILES = [
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
"""


def execute_command(command: str) -> str:
    if args.dry_run:
        print(command, file=sys.stderr)
        return ""

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

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
        title: str|None=None,
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
            ["cache-0b-1v", "local-1b-1v", "register-1b-1v", "register-branchless-1b-1v"],
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
            ["cache-0b-4v", "local-1b-4v", "register-1b-4v", "register-branchless-1b-4v"],
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
