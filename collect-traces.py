#!/usr/bin/python3

import os
import sys

import argparse
import logging
import subprocess
from typing import NewType


AVAILABLE_METRICS = {
    "inst_per_warp": "Average number of instructions executed by each warp",
    "branch_efficiency": "Ratio of non-divergent branches to total branches expressed as percentage",
    "warp_execution_efficiency": "Ratio of the average active threads per warp to the maximum number of threads per warp supported on a multiprocessor",
    "warp_nonpred_execution_efficiency": "Ratio of the average active threads per warp executing non-predicated instructions to the maximum number of threads per warp supported on a multiprocessor",
    "inst_replay_overhead": "Average number of replays for each instruction executed",
    "shared_load_transactions_per_request": "Average number of shared memory load transactions performed for each shared memory load",
    "shared_store_transactions_per_request": "Average number of shared memory store transactions performed for each shared memory store",
    "local_load_transactions_per_request": "Average number of local memory load transactions performed for each local memory load",
    "local_store_transactions_per_request": "Average number of local memory store transactions performed for each local memory store",
    "gld_transactions_per_request": "Average number of global memory load transactions performed for each global memory load.",
    "gst_transactions_per_request": "Average number of global memory store transactions performed for each global memory store",
    "shared_store_transactions": "Number of shared memory store transactions",
    "shared_load_transactions": "Number of shared memory load transactions",
    "local_load_transactions": "Number of local memory load transactions",
    "local_store_transactions": "Number of local memory store transactions",
    "gld_transactions": "Number of global memory load transactions",
    "gst_transactions": "Number of global memory store transactions",
    "sysmem_read_transactions": "Number of system memory read transactions",
    "sysmem_write_transactions": "Number of system memory write transactions",
    "l2_read_transactions": "Memory read transactions seen at L2 cache for all read requests",
    "l2_write_transactions": "Memory write transactions seen at L2 cache for all write requests",
    "global_hit_rate": "Hit rate for global loads in unified l1/tex cache. Metric value maybe wrong if malloc is used in kernel.",
    "local_hit_rate": "Hit rate for local loads and stores",
    "gld_requested_throughput": "Requested global memory load throughput",
    "gst_requested_throughput": "Requested global memory store throughput",
    "gld_throughput": "Global memory load throughput",
    "gst_throughput": "Global memory store throughput",
    "local_memory_overhead": "Ratio of local memory traffic to total memory traffic between the L1 and L2 caches expressed as percentage",
    "tex_cache_hit_rate": "Unified cache hit rate",
    "l2_tex_read_hit_rate": "Hit rate at L2 cache for all read requests from texture cache",
    "l2_tex_write_hit_rate": "Hit Rate at L2 cache for all write requests from texture cache",
    "tex_cache_throughput": "Unified cache throughput",
    "l2_tex_read_throughput": "Memory read throughput seen at L2 cache for read requests from the texture cache",
    "l2_tex_write_throughput": "Memory write throughput seen at L2 cache for write requests from the texture cache",
    "l2_read_throughput": "Memory read throughput seen at L2 cache for all read requests",
    "l2_write_throughput": "Memory write throughput seen at L2 cache for all write requests",
    "sysmem_read_throughput": "System memory read throughput",
    "sysmem_write_throughput": "System memory write throughput",
    "local_load_throughput": "Local memory load throughput",
    "local_store_throughput": "Local memory store throughput",
    "shared_load_throughput": "Shared memory load throughput",
    "shared_store_throughput": "Shared memory store throughput",
    "gld_efficiency": "Ratio of requested global memory load throughput to required global memory load throughput expressed as percentage.",
    "gst_efficiency": "Ratio of requested global memory store throughput to required global memory store throughput expressed as percentage.",
    "tex_cache_transactions": "Unified cache read transactions",
    "flop_count_dp": "Number of double-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count.",
    "flop_count_dp_add": "Number of double-precision floating-point add operations executed by non-predicated threads.",
    "flop_count_dp_fma": "Number of double-precision floating-point multiply-accumulate operations executed by non-predicated threads. Each multiply-accumulate operation contributes 1 to the count.",
    "flop_count_dp_mul": "Number of double-precision floating-point multiply operations executed by non-predicated threads.",
    "flop_count_sp": "Number of single-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count. The count does not include special operations.",
    "flop_count_sp_add": "Number of single-precision floating-point add operations executed by non-predicated threads.",
    "flop_count_sp_fma": "Number of single-precision floating-point multiply-accumulate operations executed by non-predicated threads. Each multiply-accumulate operation contributes 1 to the count.",
    "flop_count_sp_mul": "Number of single-precision floating-point multiply operations executed by non-predicated threads.",
    "flop_count_sp_special": "Number of single-precision floating-point special operations executed by non-predicated threads.",
    "inst_executed": "The number of instructions executed",
    "inst_issued": "The number of instructions issued",
    "sysmem_utilization": "The utilization level of the system memory relative to the peak utilization on a scale of 0 to 10",
    "stall_inst_fetch": "Percentage of stalls occurring because the next assembly instruction has not yet been fetched",
    "stall_exec_dependency": "Percentage of stalls occurring because an input required by the instruction is not yet available",
    "stall_memory_dependency": "Percentage of stalls occurring because a memory operation cannot be performed due to the required resources not being available or fully utilized, or because too many requests of a given type are outstanding",
    "stall_texture": "Percentage of stalls occurring because the texture sub-system is fully utilized or has too many outstanding requests",
    "stall_sync": "Percentage of stalls occurring because the warp is blocked at a __syncthreads() call",
    "stall_other": "Percentage of stalls occurring due to miscellaneous reasons",
    "stall_constant_memory_dependency": "Percentage of stalls occurring because of immediate constant cache miss",
    "stall_pipe_busy": "Percentage of stalls occurring because a compute operation cannot be performed because the compute pipeline is busy",
    "shared_efficiency": "Ratio of requested shared memory throughput to required shared memory throughput expressed as percentage",
    "inst_fp_32": "Number of single-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)",
    "inst_fp_64": "Number of double-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)",
    "inst_integer": "Number of integer instructions executed by non-predicated threads",
    "inst_bit_convert": "Number of bit-conversion instructions executed by non-predicated threads",
    "inst_control": "Number of control-flow instructions executed by non-predicated threads (jump, branch, etc.)",
    "inst_compute_ld_st": "Number of compute load/store instructions executed by non-predicated threads",
    "inst_misc": "Number of miscellaneous instructions executed by non-predicated threads",
    "inst_inter_thread_communication": "Number of inter-thread communication instructions executed by non-predicated threads",
    "issue_slots": "The number of issue slots used",
    "cf_issued": "Number of issued control-flow instructions",
    "cf_executed": "Number of executed control-flow instructions",
    "ldst_issued": "Number of issued local, global, shared and texture memory load and store instructions",
    "ldst_executed": "Number of executed local, global, shared and texture memory load and store instructions",
    "atomic_transactions": "Global memory atomic and reduction transactions",
    "atomic_transactions_per_request": "Average number of global memory atomic and reduction transactions performed for each atomic and reduction instruction",
    "l2_atomic_throughput": "Memory read throughput seen at L2 cache for atomic and reduction requests",
    "l2_atomic_transactions": "Memory read transactions seen at L2 cache for atomic and reduction requests",
    "l2_tex_read_transactions": "Memory read transactions seen at L2 cache for read requests from the texture cache",
    "stall_memory_throttle": "Percentage of stalls occurring because of memory throttle",
    "stall_not_selected": "Percentage of stalls occurring because warp was not selected",
    "l2_tex_write_transactions": "Memory write transactions seen at L2 cache for write requests from the texture cache",
    "flop_count_hp": "Number of half-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count.",
    "flop_count_hp_add": "Number of half-precision floating-point add operations executed by non-predicated threads.",
    "flop_count_hp_mul": "Number of half-precision floating-point multiply operations executed by non-predicated threads.",
    "flop_count_hp_fma": "Number of half-precision floating-point multiply-accumulate operations executed by non-predicated threads. Each multiply-accumulate operation contributes 1 to the count.",
    "inst_fp_16": "Number of half-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)",
    "sysmem_read_utilization": "The read utilization level of the system memory relative to the peak utilization on a scale of 0 to 10",
    "sysmem_write_utilization": "The write utilization level of the system memory relative to the peak utilization on a scale of 0 to 10",
    "pcie_total_data_transmitted": "Total data bytes transmitted through PCIe",
    "pcie_total_data_received": "Total data bytes received through PCIe",
    "inst_executed_global_loads": "Warp level instructions for global loads",
    "inst_executed_local_loads": "Warp level instructions for local loads",
    "inst_executed_shared_loads": "Warp level instructions for shared loads",
    "inst_executed_surface_loads": "Warp level instructions for surface loads",
    "inst_executed_global_stores": "Warp level instructions for global stores",
    "inst_executed_local_stores": "Warp level instructions for local stores",
    "inst_executed_shared_stores": "Warp level instructions for shared stores",
    "inst_executed_surface_stores": "Warp level instructions for surface stores",
    "inst_executed_global_atomics": "Warp level instructions for global atom and atom cas",
    "inst_executed_global_reductions": "Warp level instructions for global reductions",
    "inst_executed_surface_atomics": "Warp level instructions for surface atom and atom cas",
    "inst_executed_surface_reductions": "Warp level instructions for surface reductions",
    "inst_executed_shared_atomics": "Warp level shared instructions for atom and atom CAS",
    "inst_executed_tex_ops": "Warp level instructions for texture",
    "l2_global_load_bytes": "Bytes read from L2 for misses in Unified Cache for global loads",
    "l2_local_load_bytes": "Bytes read from L2 for misses in Unified Cache for local loads",
    "l2_surface_load_bytes": "Bytes read from L2 for misses in Unified Cache for surface loads",
    "l2_local_global_store_bytes": "Bytes written to L2 from Unified Cache for local and global stores. This does not include global atomics.",
    "l2_global_reduction_bytes": "Bytes written to L2 from Unified cache for global reductions",
    "l2_global_atomic_store_bytes": "Bytes written to L2 from Unified cache for global atomics (ATOM and ATOM CAS)",
    "l2_surface_store_bytes": "Bytes written to L2 from Unified Cache for surface stores. This does not include surface atomics.",
    "l2_surface_reduction_bytes": "Bytes written to L2 from Unified Cache for surface reductions",
    "l2_surface_atomic_store_bytes": "Bytes transferred between Unified Cache and L2 for surface atomics (ATOM and ATOM CAS)",
    "global_load_requests": "Total number of global load requests from Multiprocessor",
    "local_load_requests": "Total number of local load requests from Multiprocessor",
    "surface_load_requests": "Total number of surface load requests from Multiprocessor",
    "global_store_requests": "Total number of global store requests from Multiprocessor. This does not include atomic requests.",
    "local_store_requests": "Total number of local store requests from Multiprocessor",
    "surface_store_requests": "Total number of surface store requests from Multiprocessor",
    "global_atomic_requests": "Total number of global atomic(Atom and Atom CAS) requests from Multiprocessor",
    "global_reduction_requests": "Total number of global reduction requests from Multiprocessor",
    "surface_atomic_requests": "Total number of surface atomic(Atom and Atom CAS) requests from Multiprocessor",
    "surface_reduction_requests": "Total number of surface reduction requests from Multiprocessor",
    "sysmem_read_bytes": "Number of bytes read from system memory",
    "sysmem_write_bytes": "Number of bytes written to system memory",
    "l2_tex_hit_rate": "Hit rate at L2 cache for all requests from texture cache",
    "texture_load_requests": "Total number of texture Load requests from Multiprocessor",
    "unique_warps_launched": "Number of warps launched. Value is unaffected by compute preemption.",
    "sm_efficiency": "The percentage of time at least one warp is active on a specific multiprocessor",
    "achieved_occupancy": "Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor",
    "ipc": "Instructions executed per cycle",
    "issued_ipc": "Instructions issued per cycle",
    "issue_slot_utilization": "Percentage of issue slots that issued at least one instruction, averaged across all cycles",
    "eligible_warps_per_cycle": "Average number of warps that are eligible to issue per active cycle",
    "tex_utilization": "The utilization level of the unified cache relative to the peak utilization on a scale of 0 to 10",
    "l2_utilization": "The utilization level of the L2 cache relative to the peak utilization on a scale of 0 to 10",
    "shared_utilization": "The utilization level of the shared memory relative to peak utilization on a scale of 0 to 10",
    "ldst_fu_utilization": "The utilization level of the multiprocessor function units that execute shared load, shared store and constant load instructions on a scale of 0 to 10",
    "cf_fu_utilization": "The utilization level of the multiprocessor function units that execute control-flow instructions on a scale of 0 to 10",
    "special_fu_utilization": "The utilization level of the multiprocessor function units that execute sin, cos, ex2, popc, flo, and similar instructions on a scale of 0 to 10",
    "tex_fu_utilization": "The utilization level of the multiprocessor function units that execute global, local and texture memory instructions on a scale of 0 to 10",
    "single_precision_fu_utilization": "The utilization level of the multiprocessor function units that execute single-precision floating-point instructions and integer instructions on a scale of 0 to 10",
    "double_precision_fu_utilization": "The utilization level of the multiprocessor function units that execute double-precision floating-point instructions on a scale of 0 to 10",
    "flop_hp_efficiency": "Ratio of achieved to peak half-precision floating-point operations",
    "flop_sp_efficiency": "Ratio of achieved to peak single-precision floating-point operations",
    "flop_dp_efficiency": "Ratio of achieved to peak double-precision floating-point operations",
    "dram_read_transactions": "Device memory read transactions",
    "dram_write_transactions": "Device memory write transactions",
    "dram_read_throughput": "Device memory read throughput",
    "dram_write_throughput": "Device memory write throughput",
    "dram_utilization": "The utilization level of the device memory relative to the peak utilization on a scale of 0 to 10",
    "half_precision_fu_utilization": "The utilization level of the multiprocessor function units that execute 16 bit floating-point instructions on a scale of 0 to 10",
    "ecc_transactions": "Number of ECC transactions between L2 and DRAM",
    "ecc_throughput": "ECC throughput from L2 to DRAM",
    "dram_read_bytes": "Total bytes read from DRAM to L2 cache",
    "dram_write_bytes": "Total bytes written from L2 cache to DRAM",
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


NanoSeconds = NewType("NanoSeconds", int)


class Metric(str):
    def __new__(cls, metric: str):
        assert metric in AVAILABLE_METRICS.keys()
        instance = super().__new__(cls, metric)
        return instance


def parse_metrics_value(value):
    if "." in value:
        return float(value)

    try:
        return int(value)
    except:
        return value


def parse_execution_time_line(line: str) -> NanoSeconds:
    multiplier = None

    unit: str = line[-2]
    if unit == "u":
        multiplier = 1000
    elif unit == "m":
        multiplier = 1000000
    else:
        raise Exception(f"Could not parse {line}, unknown unit: {unit}.")

    value: float = float(line[:-2])

    return NanoSeconds(int(value * multiplier))


class NvprofLine:
    def __init__(self, line: str, is_execution_times: bool) -> None:
        if is_execution_times:
            self.values = [parse_execution_time_line(line.split()[1])]
        else:
            self.values = [parse_metrics_value(value) for value in line.split()[6:]]

    def __getitem__(self, arg: int):
        return self.values[arg]

    def __len__(self):
        return len(self.values)


class NvprofCommand:
    command: str

    def __init__(self, command: str, metrics: str, nvprof_path: str) -> None:
        self.command = f"sudo {nvprof_path} --print-gpu-trace "

        metrics_param = (
            [Metric(metric) for metric in metrics.split(",")]
            if len(metrics) > 0
            else []
        )
        if len(metrics_param) > 0:
            self.command += "--metrics " + ",".join(metrics_param) + " "

        self.command += command

    def __call__(self) -> list[NvprofLine]:
        output = subprocess.run(
            self.command, stderr=subprocess.PIPE, shell=True
        ).stderr.decode("utf-8")
        output = output.split("Profiling result:")[1]

        lines = [line.strip() for line in output.split("\n")]
        lines = [line for line in lines if "void" in line]

        is_execution_times = lines[0][0].isnumeric()

        return [NvprofLine(line, is_execution_times) for line in lines]


def main(args):
    if os.geteuid() != 0:
        exit("nvprof needs root access")

    if args.command is None:
        print_metrics()
        return

    nvprof_command = NvprofCommand(args.command, args.metrics, args.nvprof_path)

    outputs: list[list[NvprofLine]] = []
    for _ in range(args.repeat):
        outputs.append(nvprof_command())

    for output in outputs:
        for line in output:
            print(line[1])

    # TODO
    # Merge results with a merger (average, median, median except top and bottom)
    # Create csv
    # Detect what kind of run it was, assign the correct ec and vbw to each run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument("out_file", type=argparse.FileType("w"), help="Output file")
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
        default="",
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
