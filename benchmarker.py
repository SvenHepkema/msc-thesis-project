#!/usr/bin/python3

import itertools
import os
import sys

import argparse
import logging

import time

import inspect
import types
import subprocess
from pathlib import Path

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

NON_STANDARD_UNPACKERS = [
    "dummy",
    "old_fls_adjusted",
]

ALL_UNPACKERS = STATEFUL_UNPACKERS + OTHER_UNPACKERS

VALID_UNPACKERS = ALL_UNPACKERS + NON_STANDARD_UNPACKERS

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
ALL_N_VALS = [1, 4, 32]

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


class Stopwatch:
    def __init__(self):
        self._start_time = None

    def start(self):
        assert self._start_time is None
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        assert self._start_time is not None
        return time.perf_counter() - self._start_time


def has_root_privileges() -> bool:
    return os.geteuid() == 0


def directory_exists(path: str) -> bool:
    return Path(path).is_dir()


def get_benchmarking_functions() -> list[tuple[str, types.FunctionType]]:
    current_module = inspect.getmodule(inspect.currentframe())
    functions = inspect.getmembers(current_module, inspect.isfunction)

    benchmarking_function_prefix = "bench_"
    benchmarking_functions = filter(
        lambda x: x[0].startswith(benchmarking_function_prefix), functions
    )
    stripped_prefixes_from_name = map(
        lambda x: (x[0].replace(benchmarking_function_prefix, ""), x[1]),
        benchmarking_functions,
    )
    return list(stripped_prefixes_from_name)


def execute_command(command: str) -> str:
    if args.dry_run:
        print(command, file=sys.stderr)
        return ""

    sw = Stopwatch().start()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    execution_time_in_s = sw.stop()

    logging.info(f"Executed in {execution_time_in_s:.3f} s: {command}")

    if result.returncode != 0:
        logging.critical(f"Exited with code {result.returncode}: {command}")
        logging.critical(f"STDOUT: {result.stdout}")
        logging.critical(f"STDERR: {result.stderr}")

    return result.stdout


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

        assert unpacker in VALID_UNPACKERS
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


class BenchmarkMeasurer:
    n_timing_runs: int
    metrics: list[str]
    output_dir: str

    def __init__(self, n_timing_runs: int, metrics: list[str], output_dir: str) -> None:
        self.n_timing_runs = n_timing_runs
        self.metrics = metrics

        assert directory_exists(output_dir)
        self.output_dir = output_dir

    def measure(self, command: BenchmarkCommand) -> None:
        execute_command(
            " ".join(
                [
                    COLLECT_TRACES_PATH,
                    f"-tr {self.n_timing_runs}",
                    f"-m {','.join(self.metrics)}",
                    f"-o {os.path.join(self.output_dir, command.get_file_name())}",
                    f'-c "{command.get_as_shell_str()}"',
                ]
            )
        )


def return_fls_vbw_benches(
    experiment: str, unpackers: list[str]
) -> list[BenchmarkCommand]:
    return [
        BenchmarkCommand(
            experiment=experiment,
            unpacker=unpacker,
            patcher="none",
            n_vecs=n_vecs,
            data_generation_definition="vbw-0-32",
            input_n_vecs=args.n_input_vecs,
        )
        for unpacker in unpackers
        for n_vecs in [1, 4]
    ]


def bench_non_standard_unpackers() -> list[BenchmarkCommand]:
    return [
        BenchmarkCommand(
            experiment=experiment,
            unpacker=unpacker,
            patcher="none",
            n_vecs=n_vecs,
            n_vals=n_vals,
            data_generation_definition=data_generation_definition,
            input_n_vecs=args.n_input_vecs,
        )
        for experiment in [
            "fls_decompress",
            "fls_query",
            "fls_query_unrolled",
            "fls_compute",
        ]
        for unpacker, data_generation_definition, n_vecs, n_vals in zip(
            [
                NON_STANDARD_UNPACKERS[0],
                NON_STANDARD_UNPACKERS[0],
                NON_STANDARD_UNPACKERS[1],
            ],
            [
                "vbw-32",
                "vbw-32",
                "vbw-0-32",
            ],
            [
                1,
                4,
                1,
            ],
            [
                1,
                1,
                32,
            ],
        )
    ]


def bench_all_fls_query_vbw() -> list[BenchmarkCommand]:
    return return_fls_vbw_benches("fls_query", ALL_UNPACKERS)


def bench_all_fls_query_unrolled_vbw() -> list[BenchmarkCommand]:
    return return_fls_vbw_benches("fls_query_unrolled", ALL_UNPACKERS)


def bench_all_fls_decompress_vbw() -> list[BenchmarkCommand]:
    return return_fls_vbw_benches("fls_decompress", MAIN_UNPACKERS)


def bench_all_fls_compute_vbw() -> list[BenchmarkCommand]:
    return return_fls_vbw_benches("fls_compute", MAIN_UNPACKERS)


def bench_all_alp_query_ec() -> list[BenchmarkCommand]:
    return [
        BenchmarkCommand(
            experiment="alp_query",
            unpacker=get_main_alp_unpacker_for_n_vecs(n_vecs),
            patcher=patcher,
            n_vecs=n_vecs,
            data_generation_definition="ec-0-50",
            input_n_vecs=args.n_input_vecs,
            datatype_width=datatype_width,
        )
        for patcher in ALL_PATCHERS
        for n_vecs in [1, 4]
        for datatype_width in [32, 64]
    ]


def bench_all_alp_query_vbw() -> list[BenchmarkCommand]:
    return [
        BenchmarkCommand(
            experiment="alp_query",
            unpacker=get_main_alp_unpacker_for_n_vecs(n_vecs),
            patcher=patcher,
            n_vecs=n_vecs,
            data_generation_definition=f"vbw-0-{datatype_width}",
            input_n_vecs=args.n_input_vecs,
            datatype_width=datatype_width,
        )
        for patcher in PARALLEL_PATCHERS
        for n_vecs in [1, 4]
        for datatype_width in [32, 64]
    ]


def main(args):
    assert has_root_privileges()

    measurer = BenchmarkMeasurer(args.n_timing_runs, ALL_METRICS, args.output_dir)
    benches = args.benchmarking_function()
    for i, bench in enumerate(benches):
        logging.info(
            f"Executing benchmark {i+1}/{len(benches)}: {bench.get_as_shell_str()}"
        )
        measurer.measure(bench)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    benchmarking_functions = {func[0]: func[1] for func in get_benchmarking_functions()}
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "output_dir",
        type=str,
    )
    parser.add_argument(
        "benchmarking_function",
        type=str,
        choices=list(benchmarking_functions.keys()) + ["all"],
    )
    parser.add_argument(
        "-ntr",
        "--n-timing-runs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-niv",
        "--n-input-vecs",
        type=int,
        default=INPUT_STANDARD_N_VECS,
    )
    parser.add_argument(
        "-dr",
        "--dry-run",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
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

    if args.benchmarking_function == "all":
        args.benchmarking_function = lambda: list(
            itertools.chain.from_iterable(
                [func() for func in benchmarking_functions.values()]
            )
        )
    else:
        args.benchmarking_function = benchmarking_functions[args.benchmarking_function]

    main(args)
