#!/usr/bin/python3

import os
import sys

import typing
from collections import defaultdict
import numpy as np
import polars as pl
import time
import argparse
import logging
import subprocess
from typing import Any, NewType

NanoSeconds = NewType("NanoSeconds", int)
Metrics = list[str] | None
MeasurementsAllRuns = list[list[list[int | float]]]
MergedMeasurementsPerKernel = list[list[int | float]]

MERGE_STRATEGIES = [
    "median",
    "mean",
    "median_without_outliers",
]

AVAILABLE_METRICS = {
    "achieved_occupancy": "Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor",
    "atomic_transactions": "Global memory atomic and reduction transactions",
    "atomic_transactions_per_request": "Average number of global memory atomic and reduction transactions performed for each atomic and reduction instruction",
    "branch_efficiency": "Ratio of non-divergent branches to total branches expressed as percentage",
    "cf_executed": "Number of executed control-flow instructions",
    "cf_fu_utilization": "The utilization level of the multiprocessor function units that execute control-flow instructions on a scale of 0 to 10",
    "cf_issued": "Number of issued control-flow instructions",
    "double_precision_fu_utilization": "The utilization level of the multiprocessor function units that execute double-precision floating-point instructions on a scale of 0 to 10",
    "dram_read_bytes": "Total bytes read from DRAM to L2 cache",
    "dram_read_throughput": "Device memory read throughput",
    "dram_read_transactions": "Device memory read transactions",
    "dram_utilization": "The utilization level of the device memory relative to the peak utilization on a scale of 0 to 10",
    "dram_write_bytes": "Total bytes written from L2 cache to DRAM",
    "dram_write_throughput": "Device memory write throughput",
    "dram_write_transactions": "Device memory write transactions",
    "ecc_throughput": "ECC throughput from L2 to DRAM",
    "ecc_transactions": "Number of ECC transactions between L2 and DRAM",
    "eligible_warps_per_cycle": "Average number of warps that are eligible to issue per active cycle",
    "flop_count_dp_add": "Number of double-precision floating-point add operations executed by non-predicated threads.",
    "flop_count_dp_fma": "Number of double-precision floating-point multiply-accumulate operations executed by non-predicated threads. Each multiply-accumulate operation contributes 1 to the count.",
    "flop_count_dp_mul": "Number of double-precision floating-point multiply operations executed by non-predicated threads.",
    "flop_count_dp": "Number of double-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count.",
    "flop_count_hp_add": "Number of half-precision floating-point add operations executed by non-predicated threads.",
    "flop_count_hp_fma": "Number of half-precision floating-point multiply-accumulate operations executed by non-predicated threads. Each multiply-accumulate operation contributes 1 to the count.",
    "flop_count_hp_mul": "Number of half-precision floating-point multiply operations executed by non-predicated threads.",
    "flop_count_hp": "Number of half-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count.",
    "flop_count_sp_add": "Number of single-precision floating-point add operations executed by non-predicated threads.",
    "flop_count_sp_fma": "Number of single-precision floating-point multiply-accumulate operations executed by non-predicated threads. Each multiply-accumulate operation contributes 1 to the count.",
    "flop_count_sp_mul": "Number of single-precision floating-point multiply operations executed by non-predicated threads.",
    "flop_count_sp": "Number of single-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count. The count does not include special operations.",
    "flop_count_sp_special": "Number of single-precision floating-point special operations executed by non-predicated threads.",
    "flop_dp_efficiency": "Ratio of achieved to peak double-precision floating-point operations",
    "flop_hp_efficiency": "Ratio of achieved to peak half-precision floating-point operations",
    "flop_sp_efficiency": "Ratio of achieved to peak single-precision floating-point operations",
    "gld_efficiency": "Ratio of requested global memory load throughput to required global memory load throughput expressed as percentage.",
    "gld_requested_throughput": "Requested global memory load throughput",
    "gld_throughput": "Global memory load throughput",
    "gld_transactions": "Number of global memory load transactions",
    "gld_transactions_per_request": "Average number of global memory load transactions performed for each global memory load.",
    "global_atomic_requests": "Total number of global atomic(Atom and Atom CAS) requests from Multiprocessor",
    "global_hit_rate": "Hit rate for global loads in unified l1/tex cache. Metric value maybe wrong if malloc is used in kernel.",
    "global_load_requests": "Total number of global load requests from Multiprocessor",
    "global_reduction_requests": "Total number of global reduction requests from Multiprocessor",
    "global_store_requests": "Total number of global store requests from Multiprocessor. This does not include atomic requests.",
    "gst_efficiency": "Ratio of requested global memory store throughput to required global memory store throughput expressed as percentage.",
    "gst_requested_throughput": "Requested global memory store throughput",
    "gst_throughput": "Global memory store throughput",
    "gst_transactions": "Number of global memory store transactions",
    "gst_transactions_per_request": "Average number of global memory store transactions performed for each global memory store",
    "half_precision_fu_utilization": "The utilization level of the multiprocessor function units that execute 16 bit floating-point instructions on a scale of 0 to 10",
    "inst_bit_convert": "Number of bit-conversion instructions executed by non-predicated threads",
    "inst_compute_ld_st": "Number of compute load/store instructions executed by non-predicated threads",
    "inst_control": "Number of control-flow instructions executed by non-predicated threads (jump, branch, etc.)",
    "inst_executed_global_atomics": "Warp level instructions for global atom and atom cas",
    "inst_executed_global_loads": "Warp level instructions for global loads",
    "inst_executed_global_reductions": "Warp level instructions for global reductions",
    "inst_executed_global_stores": "Warp level instructions for global stores",
    "inst_executed_local_loads": "Warp level instructions for local loads",
    "inst_executed_local_stores": "Warp level instructions for local stores",
    "inst_executed_shared_atomics": "Warp level shared instructions for atom and atom CAS",
    "inst_executed_shared_loads": "Warp level instructions for shared loads",
    "inst_executed_shared_stores": "Warp level instructions for shared stores",
    "inst_executed_surface_atomics": "Warp level instructions for surface atom and atom cas",
    "inst_executed_surface_loads": "Warp level instructions for surface loads",
    "inst_executed_surface_reductions": "Warp level instructions for surface reductions",
    "inst_executed_surface_stores": "Warp level instructions for surface stores",
    "inst_executed_tex_ops": "Warp level instructions for texture",
    "inst_executed": "The number of instructions executed",
    "inst_fp_16": "Number of half-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)",
    "inst_fp_32": "Number of single-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)",
    "inst_fp_64": "Number of double-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)",
    "inst_integer": "Number of integer instructions executed by non-predicated threads",
    "inst_inter_thread_communication": "Number of inter-thread communication instructions executed by non-predicated threads",
    "inst_issued": "The number of instructions issued",
    "inst_misc": "Number of miscellaneous instructions executed by non-predicated threads",
    "inst_per_warp": "Average number of instructions executed by each warp",
    "inst_replay_overhead": "Average number of replays for each instruction executed",
    "ipc": "Instructions executed per cycle",
    "issued_ipc": "Instructions issued per cycle",
    "issue_slots": "The number of issue slots used",
    "issue_slot_utilization": "Percentage of issue slots that issued at least one instruction, averaged across all cycles",
    "l2_atomic_throughput": "Memory read throughput seen at L2 cache for atomic and reduction requests",
    "l2_atomic_transactions": "Memory read transactions seen at L2 cache for atomic and reduction requests",
    "l2_global_atomic_store_bytes": "Bytes written to L2 from Unified cache for global atomics (ATOM and ATOM CAS)",
    "l2_global_load_bytes": "Bytes read from L2 for misses in Unified Cache for global loads",
    "l2_global_reduction_bytes": "Bytes written to L2 from Unified cache for global reductions",
    "l2_local_global_store_bytes": "Bytes written to L2 from Unified Cache for local and global stores. This does not include global atomics.",
    "l2_local_load_bytes": "Bytes read from L2 for misses in Unified Cache for local loads",
    "l2_read_throughput": "Memory read throughput seen at L2 cache for all read requests",
    "l2_read_transactions": "Memory read transactions seen at L2 cache for all read requests",
    "l2_surface_atomic_store_bytes": "Bytes transferred between Unified Cache and L2 for surface atomics (ATOM and ATOM CAS)",
    "l2_surface_load_bytes": "Bytes read from L2 for misses in Unified Cache for surface loads",
    "l2_surface_reduction_bytes": "Bytes written to L2 from Unified Cache for surface reductions",
    "l2_surface_store_bytes": "Bytes written to L2 from Unified Cache for surface stores. This does not include surface atomics.",
    "l2_tex_hit_rate": "Hit rate at L2 cache for all requests from texture cache",
    "l2_tex_read_hit_rate": "Hit rate at L2 cache for all read requests from texture cache",
    "l2_tex_read_throughput": "Memory read throughput seen at L2 cache for read requests from the texture cache",
    "l2_tex_read_transactions": "Memory read transactions seen at L2 cache for read requests from the texture cache",
    "l2_tex_write_hit_rate": "Hit Rate at L2 cache for all write requests from texture cache",
    "l2_tex_write_throughput": "Memory write throughput seen at L2 cache for write requests from the texture cache",
    "l2_tex_write_transactions": "Memory write transactions seen at L2 cache for write requests from the texture cache",
    "l2_utilization": "The utilization level of the L2 cache relative to the peak utilization on a scale of 0 to 10",
    "l2_write_throughput": "Memory write throughput seen at L2 cache for all write requests",
    "l2_write_transactions": "Memory write transactions seen at L2 cache for all write requests",
    "ldst_executed": "Number of executed local, global, shared and texture memory load and store instructions",
    "ldst_fu_utilization": "The utilization level of the multiprocessor function units that execute shared load, shared store and constant load instructions on a scale of 0 to 10",
    "ldst_issued": "Number of issued local, global, shared and texture memory load and store instructions",
    "local_hit_rate": "Hit rate for local loads and stores",
    "local_load_requests": "Total number of local load requests from Multiprocessor",
    "local_load_throughput": "Local memory load throughput",
    "local_load_transactions": "Number of local memory load transactions",
    "local_load_transactions_per_request": "Average number of local memory load transactions performed for each local memory load",
    "local_memory_overhead": "Ratio of local memory traffic to total memory traffic between the L1 and L2 caches expressed as percentage",
    "local_store_requests": "Total number of local store requests from Multiprocessor",
    "local_store_throughput": "Local memory store throughput",
    "local_store_transactions": "Number of local memory store transactions",
    "local_store_transactions_per_request": "Average number of local memory store transactions performed for each local memory store",
    "pcie_total_data_received": "Total data bytes received through PCIe",
    "pcie_total_data_transmitted": "Total data bytes transmitted through PCIe",
    "shared_efficiency": "Ratio of requested shared memory throughput to required shared memory throughput expressed as percentage",
    "shared_load_throughput": "Shared memory load throughput",
    "shared_load_transactions": "Number of shared memory load transactions",
    "shared_load_transactions_per_request": "Average number of shared memory load transactions performed for each shared memory load",
    "shared_store_throughput": "Shared memory store throughput",
    "shared_store_transactions": "Number of shared memory store transactions",
    "shared_store_transactions_per_request": "Average number of shared memory store transactions performed for each shared memory store",
    "shared_utilization": "The utilization level of the shared memory relative to peak utilization on a scale of 0 to 10",
    "single_precision_fu_utilization": "The utilization level of the multiprocessor function units that execute single-precision floating-point instructions and integer instructions on a scale of 0 to 10",
    "sm_efficiency": "The percentage of time at least one warp is active on a specific multiprocessor",
    "special_fu_utilization": "The utilization level of the multiprocessor function units that execute sin, cos, ex2, popc, flo, and similar instructions on a scale of 0 to 10",
    "stall_constant_memory_dependency": "Percentage of stalls occurring because of immediate constant cache miss",
    "stall_exec_dependency": "Percentage of stalls occurring because an input required by the instruction is not yet available",
    "stall_inst_fetch": "Percentage of stalls occurring because the next assembly instruction has not yet been fetched",
    "stall_memory_dependency": "Percentage of stalls occurring because a memory operation cannot be performed due to the required resources not being available or fully utilized, or because too many requests of a given type are outstanding",
    "stall_memory_throttle": "Percentage of stalls occurring because of memory throttle",
    "stall_not_selected": "Percentage of stalls occurring because warp was not selected",
    "stall_other": "Percentage of stalls occurring due to miscellaneous reasons",
    "stall_pipe_busy": "Percentage of stalls occurring because a compute operation cannot be performed because the compute pipeline is busy",
    "stall_sync": "Percentage of stalls occurring because the warp is blocked at a __syncthreads() call",
    "stall_texture": "Percentage of stalls occurring because the texture sub-system is fully utilized or has too many outstanding requests",
    "surface_atomic_requests": "Total number of surface atomic(Atom and Atom CAS) requests from Multiprocessor",
    "surface_load_requests": "Total number of surface load requests from Multiprocessor",
    "surface_reduction_requests": "Total number of surface reduction requests from Multiprocessor",
    "surface_store_requests": "Total number of surface store requests from Multiprocessor",
    "sysmem_read_bytes": "Number of bytes read from system memory",
    "sysmem_read_throughput": "System memory read throughput",
    "sysmem_read_transactions": "Number of system memory read transactions",
    "sysmem_read_utilization": "The read utilization level of the system memory relative to the peak utilization on a scale of 0 to 10",
    "sysmem_utilization": "The utilization level of the system memory relative to the peak utilization on a scale of 0 to 10",
    "sysmem_write_bytes": "Number of bytes written to system memory",
    "sysmem_write_throughput": "System memory write throughput",
    "sysmem_write_transactions": "Number of system memory write transactions",
    "sysmem_write_utilization": "The write utilization level of the system memory relative to the peak utilization on a scale of 0 to 10",
    "tex_cache_hit_rate": "Unified cache hit rate",
    "tex_cache_throughput": "Unified cache throughput",
    "tex_cache_transactions": "Unified cache read transactions",
    "tex_fu_utilization": "The utilization level of the multiprocessor function units that execute global, local and texture memory instructions on a scale of 0 to 10",
    "texture_load_requests": "Total number of texture Load requests from Multiprocessor",
    "tex_utilization": "The utilization level of the unified cache relative to the peak utilization on a scale of 0 to 10",
    "unique_warps_launched": "Number of warps launched. Value is unaffected by compute preemption.",
    "warp_execution_efficiency": "Ratio of the average active threads per warp to the maximum number of threads per warp supported on a multiprocessor",
    "warp_nonpred_execution_efficiency": "Ratio of the average active threads per warp executing non-predicated instructions to the maximum number of threads per warp supported on a multiprocessor",
}


def print_metrics() -> None:
    max_length_str = 0
    keys = []
    for metric in AVAILABLE_METRICS:
        max_length_str = max(max_length_str, len(metric))
        keys.append(metric)

    keys.sort()

    for metric in keys:
        print(metric.ljust(max_length_str), " : ", AVAILABLE_METRICS[metric])


def parse_metrics_value(value_str: str) -> float | int | None:
    if "." in value_str:
        return float(value_str)

    try:
        return int(value_str)
    except:
        logging.warn(f"Could not parse this value {value_str}")


def parse_execution_time_value(value_str: str) -> NanoSeconds:
    multiplier = None

    unit: str = value_str[-2]
    if unit == "u":
        multiplier = 1000
    elif unit == "m":
        multiplier = 1000000
    else:
        raise Exception(f"Could not parse {value_str}, unknown unit: {unit}.")

    value_float: float = float(value_str[:-2])

    return NanoSeconds(int(value_float * multiplier))


class NvprofLine:
    values: list[float | int]

    def __init__(self, line: str, is_execution_times: bool) -> None:
        if is_execution_times:
            self.values = [parse_execution_time_value(line.split()[1])]
        else:
            self.values = []
            for value in line.split()[6:]:
                parsed_value = parse_metrics_value(value)
                if parsed_value is not None:
                    self.values.append(parsed_value)


def parse_metrics_arg(arg: str | None) -> list[str] | None:
    if arg is None:
        return None

    parsed_metrics = []
    for metric in arg.split(","):
        if metric not in AVAILABLE_METRICS.keys():
            logging.warn(f"{metric} was not recognized as a metric")
            time.sleep(0.5)

        parsed_metrics.append(metric)

    return parsed_metrics


def convert_metrics_to_parameter(metrics: list[str]) -> str:
    return "--metrics " + ",".join(metrics) + " "


class NvprofCommand:
    command: str

    def __init__(
        self, command: str, metrics: list[str] | None, nvprof_path: str
    ) -> None:
        self.command = f"{nvprof_path} --print-gpu-trace "

        if metrics is not None:
            self.command = (
                "sudo " + self.command + convert_metrics_to_parameter(metrics)
            )

        self.command += command

    def __call__(self) -> list[NvprofLine]:
        logging.info(f"Executing: {self.command}")
        result = subprocess.run(
            self.command, stderr=subprocess.PIPE, shell=True
        )

        if result.returncode != 0:
            logging.critical("")
            logging.critical("")
            logging.critical(f"FAILED WITH CODE {result.returncode} : {self.command} ")
            logging.critical("")
            logging.critical("")

        output = result.stderr.decode("utf-8")
        output = output.split("Profiling result:")[1]

        lines = [line.strip() for line in output.split("\n")]
        lines = [line for line in lines if "void" in line]

        is_execution_times = lines[0][0].isnumeric()

        return [NvprofLine(line, is_execution_times) for line in lines]


def merge_metrics_values(values: list[Any], merge_strategy: str) -> Any:
    assert len(values) > 0
    assert not isinstance(values[0], str)

    merged_value = 0

    if merge_strategy == MERGE_STRATEGIES[0]:
        merged_value = np.mean(values)
    if merge_strategy == MERGE_STRATEGIES[1]:
        merged_value = np.median(values)
    if merge_strategy == MERGE_STRATEGIES[2]:
        merged_value = np.mean(sorted(values)[1:-1])

    if isinstance(values[0], int):
        merged_value = int(merged_value)

    return merged_value


def collect_measurements(command: NvprofCommand, repeat: int) -> MeasurementsAllRuns:
    measurements_all_runs: MeasurementsAllRuns = []
    for _ in range(repeat):
        nvprof_output = command()

        measurements_values_run = []
        for nvprof_line in nvprof_output:
            measurements_values_run.append(nvprof_line.values)

        measurements_all_runs.append(measurements_values_run)

    return measurements_all_runs


def merge_measurements(
    measurements_all_runs: MeasurementsAllRuns, metrics: Metrics, merge_strategy: str
) -> MergedMeasurementsPerKernel:
    n_metrics = len(metrics) if metrics else 1
    n_kernels = len(measurements_all_runs[0])
    merged_measurements_values: MergedMeasurementsPerKernel = []

    for k in range(n_kernels):
        kernel_measurements = []
        for m in range(n_metrics):
            values: list[int | float] = []
            for measurements_run in measurements_all_runs:
                values.append(measurements_run[k][m])
            kernel_measurements.append(merge_metrics_values(values, merge_strategy))
        merged_measurements_values.append(kernel_measurements)

    return merged_measurements_values


def create_df_from_measurements(
    metrics: Metrics,
    measurements: MergedMeasurementsPerKernel,
) -> pl.DataFrame:
    if metrics is None:
        metrics = ["execution_time"]

    n_kernels = len(measurements)
    table = defaultdict(list)
    table["kernel_id"] = list(range(n_kernels))

    for i, metric in enumerate(metrics):
        for k in range(n_kernels):
            table[metric].append(measurements[k][i])

    return pl.DataFrame(table)


class ExecutableProperties:
    datagen_type: str | None
    datagen_params: list[int] | None

    def __init__(self, exec_str: str) -> None:
        split_exec_str = exec_str.split()
        data_param = split_exec_str[5]

        if "ec" in data_param:
            self.datagen_type = "ec"
        elif "vbw" in data_param:
            self.datagen_type = "vbw"
        else:
            self.datagen_type = None

        if self.datagen_type != None:
            data_param_split = data_param.split("-")
            if len(data_param_split) == 2:
                self.datagen_params = list(range(int(data_param_split[1]) + 1))
            if len(data_param_split) == 3:
                self.datagen_params = list(
                    range(int(data_param_split[1]), int(data_param_split[2]) + 1)
                )


def apply_properties_to_df(
    properties: ExecutableProperties, df: pl.DataFrame
) -> pl.DataFrame:
    if properties.datagen_type is None:
        return df

    df = df.with_columns(
        pl.Series(name=properties.datagen_type, values=properties.datagen_params)
    )

    return df


def measure_command(
    nvprof_command: NvprofCommand, metrics: Metrics, repeat: int, merge_strategy: str
) -> pl.DataFrame:
    all_measurements = collect_measurements(nvprof_command, repeat)
    merged_measurements = merge_measurements(all_measurements, metrics, merge_strategy)
    return create_df_from_measurements(metrics, merged_measurements)


def write_results_to_sink(command: str, df: pl.DataFrame, sink: typing.TextIO) -> None:
    print(command, file=sink)
    df.write_csv(sink)


def main(args):
    if args.command is None:
        print_metrics()
        return

    metrics = parse_metrics_arg(args.metrics)
    if metrics != None and os.geteuid() != 0:
        exit("nvprof needs root access ")

    nvprof_command = NvprofCommand(args.command, metrics, args.nvprof_path)
    df = measure_command(nvprof_command, metrics, args.repeat, args.merge_strategy)

    if args.timing_runs is not None and metrics is not None:
        timing_command = NvprofCommand(args.command, None, args.nvprof_path)
        timing_df = measure_command(
            timing_command, None, args.timing_runs, args.merge_strategy
        )
        df = df.with_columns(timing_df["execution_time"])

    exec_properties = ExecutableProperties(args.command)
    df = apply_properties_to_df(exec_properties, df)

    write_results_to_sink(args.command, df, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "-o",
        "--out",
        type=argparse.FileType("w"), 
        help="Out file",
        default=sys.stdout,
    )
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        help="Number of times to repeat each run.",
        default=1,
    )
    parser.add_argument(
        "-c",
        "--command",
        type=str,
        help="Command to execute",
        default=None,
    )
    parser.add_argument(
        "-m",
        "--metrics",
        type=str,
        help="Metrics, comma separated",
        default=None,
    )
    parser.add_argument(
        "-tr",
        "--timing-runs",
        type=int,
        help="Adds timing runs if metrics are enabled, will add an extra column",
        default=None,
    )
    parser.add_argument(
        "-ms",
        "--merge-strategy",
        type=str,
        help="How values from multiple runs are merged",
        choices=MERGE_STRATEGIES,
        default=MERGE_STRATEGIES[0],
    )
    parser.add_argument(
        "--nvprof-path",
        type=str,
        help="Command to execute",
        default="/usr/local/cuda-12.5/bin/nvprof",
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

    main(args)
