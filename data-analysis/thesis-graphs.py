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

# ==========================
# ==========================
# COPIED FROM Benchmarker.py
# ==========================

INPUT_STANDARD_N_VECS = 1024 * 100

EXECUTABLE_PATH = "./bin/executable"
COLLECT_TRACES_PATH = "./collect-traces.py"

STATEFUL_UNPACKERS = ["stateful-cache"] + [
    f"stateful-{buffer_type}-{buffer_size}"
    for buffer_type in ["local", "register", "register-branchless"]
    for buffer_size in [1, 2, 4]
]
MAIN_STATEFUL_UNPACKER = "stateful-register-branchless-2"
assert MAIN_STATEFUL_UNPACKER in STATEFUL_UNPACKERS

OTHER_UNPACKERS = [
    "stateless",
    "stateless_branchless",
    "stateful_branchless",
]

ALL_UNPACKERS = STATEFUL_UNPACKERS + OTHER_UNPACKERS

MAIN_UNPACKERS = [MAIN_STATEFUL_UNPACKER] + OTHER_UNPACKERS


def get_main_alp_unpacker_for_n_vecs(n_vecs: int) -> str:
    return "stateful_branchless" if n_vecs == 1 else MAIN_STATEFUL_UNPACKER


NO_PATCHER = ["none"]
NON_PARALLEL_PATCHERS = ["stateless", "stateful"]
PARALLEL_PATCHERS = [
    "naive",
    "naive_branchless",
    "prefetch_position",
    "prefetch_all",
    "prefetch_all_branchless",
]

ALL_PATCHERS_OPTIONS = NO_PATCHER + NON_PARALLEL_PATCHERS + PARALLEL_PATCHERS
ALL_PATCHERS = NON_PARALLEL_PATCHERS + PARALLEL_PATCHERS

ALL_EXPERIMENTS = [
    "fls_query",
    "fls_query_unrolled",
    "fls_decompress",
    "fls_compute",
    "alp_decompress",
    "alp_query",
]

ALL_DATATYPE_WIDTHS = [8, 16, 32, 64]

ALL_N_VECS = [1, 4]
ALL_N_VALS = [1, 4]

ALL_DATA_GENERATION_TYPES = ["none", "ec", "vbw"]

ALL_METRICS = [
    "global_load_requests",
    "global_hit_rate",
    "l2_tex_hit_rate",
    "inst_issued",
    "stall_memory_dependency",
    "stall_memory_throttle",
    "dram_read_bytes",
    "dram_write_bytes",
    "ipc",
]


class BenchmarkCommand:
    experiment: str
    unpacker: str
    patcher: str
    n_vecs: int
    n_vals: int
    input_n_vecs: int
    data_generation_definition: str
    data_name: str
    datatype_width: int

    def __init__(
        self,
        experiment: str,
        unpacker: str,
        patcher: str,
        n_vecs: int,
        data_generation_definition: str,
        input_n_vecs: int,
        data_name: str = "random",
        datatype_width: int = 32,
        n_vals: int = 1,
    ) -> None:
        assert experiment in ALL_EXPERIMENTS
        self.experiment = experiment

        assert unpacker in ALL_UNPACKERS
        self.unpacker = unpacker

        assert patcher in ALL_PATCHERS_OPTIONS
        self.patcher = patcher

        assert n_vecs in ALL_N_VECS
        self.n_vecs = n_vecs

        assert n_vals in ALL_N_VALS
        self.n_vals = n_vals

        assert input_n_vecs > 0
        self.input_n_vecs = input_n_vecs

        assert any([x in data_generation_definition for x in ALL_DATA_GENERATION_TYPES])
        self.data_generation_definition = data_generation_definition

        # 'index' 'random' or filename
        self.data_name = data_name

        assert datatype_width in ALL_DATATYPE_WIDTHS
        self.datatype_width = datatype_width

    def get_as_shell_str(self) -> str:
        return " ".join(
            [
                EXECUTABLE_PATH,
                f"{self.experiment} {self.unpacker} {self.patcher}",
                f"{self.n_vecs} {self.n_vals} {self.datatype_width}",
                f"{self.data_name} {self.data_generation_definition}",
                f"{self.input_n_vecs} 0",
            ]
        )

    def get_file_name(self) -> str:
        return self.get_as_shell_str()[len(EXECUTABLE_PATH) + 1 :].replace(" ", "_")


# ==========================
# ==========================

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


class DataSource:
    file: BenchmarkCommand
    label: str
    color: int

    def __init__(
        self,
        file: BenchmarkCommand,
        label: str,
        color: int,
    ) -> None:
        self.file = file
        self.label = label
        self.color = color


class GraphDefintion:
    sources: list[DataSource]
    out: str
    title: str | None

    def __init__(
        self,
        sources: list[DataSource],
        out: str,
        title: str | None = None,
    ) -> None:
        self.sources = sources
        self.out = out
        self.title = title

    @property
    def file_names(self) -> list[str]:
        return [source.file.get_file_name() for source in self.sources]

    @property
    def labels(self) -> list[str]:
        return [source.label for source in self.sources]

    @property
    def colors(self) -> list[int]:
        return [source.color for source in self.sources]


def inner_plot_main_unpackers(input_dir: str, output_dir: str, experiment: str, yamv: int):
    graphs = [
        GraphDefintion(
            [
                DataSource(
                    BenchmarkCommand(
                        experiment=experiment,
                        unpacker=unpacker,
                        patcher=NO_PATCHER[0],
                        n_vecs=n_vecs,
                        data_generation_definition="vbw-0-32",
                        input_n_vecs=INPUT_STANDARD_N_VECS,
                    ),
                    label,
                    color,
                )
                for unpacker, label, color in zip(
                    [
                        "stateless",
                        "stateful-register-branchless-2",
                        "stateless_branchless",
                        "stateful_branchless",
                    ],
                    [
                        f"stateless-{n_vecs}v",
                        f"stateful-{n_vecs}v",
                        f"stateless-branchless-{n_vecs}v",
                        f"stateful-branchless-{n_vecs}v",
                    ],
                    range(0, 4),
                )
            ],
            f"unpackers-{experiment}-{n_vecs}v",
        )
        for n_vecs in [1, 4]
    ]

    for graph in graphs:
        execute_command(
            " ".join(
                [
                    f"{GRAPHER_PATH}",
                    f'{":".join([os.path.join(input_dir, file) for file in graph.file_names])}',
                    f"scatter",
                    f"execution_time",
                    f'-l {":".join(graph.labels)}',
                    f"-hl 2504880",
                    f'-hll "Normal execution"',
                    f"-lp upper-left",
                    f"-yamv {yamv}",
                    f'-c {":".join(map(str, graph.colors))}',
                    f'-o {os.path.join(output_dir,f"{graph.out}.eps")}',
                ]
            )
        )


def plot_main_unpackers_query(input_dir: str, output_dir: str):
    inner_plot_main_unpackers(input_dir, output_dir, "fls_query", 3000)


def plot_main_unpackers_query_unrolled(input_dir: str, output_dir: str):
    inner_plot_main_unpackers(input_dir, output_dir, "fls_query_unrolled", 3000)


def plot_main_unpackers_decompress(input_dir: str, output_dir: str):
    inner_plot_main_unpackers(input_dir, output_dir, "fls_decompress", 6000)


def plot_main_unpackers_compute(input_dir: str, output_dir: str):
    inner_plot_main_unpackers(input_dir, output_dir, "fls_compute", 6000)


def inner_plot_stateful_unpackers(input_dir: str, output_dir: str, experiment: str):
    graphs = [
        GraphDefintion(
            [
                DataSource(
                    BenchmarkCommand(
                        experiment=experiment,
                        unpacker=(
                            f"stateful-{buffer_type}-{buffer_size}"
                            if buffer_type != "cache"
                            else "stateful-cache"
                        ),
                        patcher=NO_PATCHER[0],
                        n_vecs=n_vecs,
                        data_generation_definition="vbw-0-32",
                        input_n_vecs=INPUT_STANDARD_N_VECS,
                    ),
                    (
                        f"{buffer_type}-{buffer_size}b-{n_vecs}v"
                        if buffer_type != "cache"
                        else f"cache-{n_vecs}v"
                    ),
                    color,
                )
                for buffer_type, color in zip(
                    [
                        "cache",
                        "local",
                        "register",
                        "register-branchless",
                    ],
                    range(0, 4),
                )
                if buffer_type != "cache" or buffer_size == 1
            ],
            f"unpackers-{experiment}-{n_vecs}v-{buffer_size}b",
        )
        for buffer_size in [1, 2, 4]
        for n_vecs in [1, 4]
    ]

    for graph in graphs:
        execute_command(
            " ".join(
                [
                    f"{GRAPHER_PATH}",
                    f'{":".join([os.path.join(input_dir, file) for file in graph.file_names])}',
                    f"scatter",
                    f"execution_time",
                    f'-l {":".join(graph.labels)}',
                    f"-hl 2504880",
                    f'-hll "Normal execution"',
                    f"-lp upper-left",
                    f"-yamv 3000",
                    f'-c {":".join(map(str, graph.colors))}',
                    f'-o {os.path.join(output_dir,f"{graph.out}.eps")}',
                ]
            )
        )


def plot_stateful_unpackers_query(input_dir: str, output_dir: str):
    inner_plot_stateful_unpackers(input_dir, output_dir, "fls_query")


def plot_patchers_query(input_dir: str, output_dir: str):
    def get_graphs_for_patchers(
        patchers: list[str], out: str, colors_offset: int
    ) -> list[GraphDefintion]:
        return [
            GraphDefintion(
                [
                    DataSource(
                        BenchmarkCommand(
                            experiment="alp_query",
                            unpacker=(
                                "stateful_branchless"
                                if n_vecs == 1
                                else "stateful-register-branchless-2"
                            ),
                            patcher=patcher,
                            n_vecs=n_vecs,
                            data_generation_definition="ec-0-50",
                            input_n_vecs=INPUT_STANDARD_N_VECS,
                        ),
                        f"{patcher}-{n_vecs}v",
                        color,
                    )
                    for patcher, color in zip(
                        patchers, range(colors_offset, colors_offset + len(patchers))
                    )
                ],
                out=f"{out}-v{n_vecs}",
            )
            for n_vecs in [1, 4]
        ]

    nonparallel_patchers = (
        NON_PARALLEL_PATCHERS,
        "nonparallel-exception-patchers",
        25000,
        0,
    )
    parallel_patchers = (
        PARALLEL_PATCHERS[:2],
        "parallel-exception-patchers",
        4000,
        len(nonparallel_patchers[0]),
    )
    prefetch_parallel_patchers = (
        PARALLEL_PATCHERS[2:],
        "prefetch-parallel-exception-patchers",
        4000,
        len(nonparallel_patchers[0]) + len(parallel_patchers[0]),
    )

    for patcher_set, out, yamv, color_offset in [
        nonparallel_patchers,
        parallel_patchers,
        prefetch_parallel_patchers,
    ]:
        for graph in get_graphs_for_patchers(patcher_set, out, color_offset):
            execute_command(
                " ".join(
                    [
                        f"{GRAPHER_PATH}",
                        f'{":".join([os.path.join(input_dir, file) for file in graph.file_names])}',
                        f"scatter",
                        f"execution_time",
                        f'-l {":".join(graph.labels)}',
                        f"-hl 2504880",
                        f'-hll "Normal execution"',
                        f"-lp lower-right",
                        f"-yamv {yamv}",
                        f'-c {":".join(map(str, graph.colors))}',
                        f'-o {os.path.join(output_dir,f"{graph.out}.eps")}',
                    ]
                )
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
