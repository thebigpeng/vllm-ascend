#!/usr/bin/env python3
"""Offline analyzer for ZBAL MoE dispatch/combine kernel performance.

Takes the profiling output of vllm-ascend's torch profiler (e.g. `--profile`
with torch_profiler_dir set, collected by
vllm_ascend/profiler/torch_npu_profiler.py) and reports the overall execution
performance of the ZBAL MoE dispatch & combine kernels.

Pipeline (two stages):
  1. Analysis: each trace directory under the given roots (one per rank,
     e.g. `<worker_name>.msprof/PROF_*`) is fed to
     `torch_npu.profiler.profiler.analyse(..., export_type="text")`, which
     generates kernel_details.csv from the raw msprof data. analyse only
     discovers PROF_* dirs one level below its input, so every trace dir is
     analysed one by one. Use --no-analyse to skip this stage when the csv
     files already exist (or when running in an environment without NPU).
  2. Parsing: kernel_details.csv files are matched and aggregated.

The kernel matching and aggregation logic mirrors zbal's test-side analyzer
(membfabric_hybrid/app/zbal/test/python/operators/perf_analyze.py):
  - kernels are matched by "Name"-column prefix in kernel_details.csv
    (prefix match because msprof may append suffixes to kernel names);
  - raw (sum, count) values are aggregated across ALL ranks / trace dirs
    before computing means, which avoids the mean-of-means bug.

ZBAL MoE kernels covered (AICore kernel symbol names, grouped by phase):
  - normal (prefill) path:    notify_dispatch, dispatch_layout,
                               dispatch_normal, combine_normal
  - low-latency (decode) path: dispatch_low_latency, combine_low_latency

Usage:
    python tools/zbal_moe_perf_analyze.py <profiling_dir> [<dir2> ...]
    python tools/zbal_moe_perf_analyze.py <dir> --no-analyse   # csv already there
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

# kernel symbol name -> MoE phase ("dispatch" or "combine")
ZBAL_MOE_KERNELS: dict[str, str] = {
    # normal (DeepEP-style prefill) path
    "notify_dispatch": "dispatch",
    "dispatch_layout": "dispatch",
    "dispatch_normal": "dispatch",
    "combine_normal": "combine",
    # low-latency (decode) path
    "dispatch_low_latency": "dispatch",
    "combine_low_latency": "combine",
}

KERNEL_ORDER = list(ZBAL_MOE_KERNELS)


def parse_csv(filepath: str) -> dict[str, list]:
    """Parse one kernel_details.csv, return {kernel: [total_us, count]}.

    The Duration column is located by header name so that column reordering
    across CANN versions is tolerated (same trick as zbal's perf_analyze.py).
    """
    stats: dict[str, list] = {kw: [0.0, 0] for kw in KERNEL_ORDER}
    try:
        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            dur_idx = name_idx = None
            for i, col in enumerate(header):
                col = col.strip()
                if col == "Duration(us)":
                    dur_idx = i
                elif col == "Name":
                    name_idx = i
            if dur_idx is None or name_idx is None:
                return stats
            for row in reader:
                if len(row) <= max(dur_idx, name_idx):
                    continue
                name = row[name_idx]
                for kw in KERNEL_ORDER:
                    if name.startswith(kw):
                        try:
                            stats[kw][0] += float(row[dur_idx])
                            stats[kw][1] += 1
                        except ValueError:
                            continue
                        break
    except OSError:
        pass
    return stats


def find_kernel_csvs(roots: list[str]) -> list[str]:
    """Recursively find kernel_details.csv under the given profiling roots."""
    csvs = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn == "kernel_details.csv":
                    csvs.append(os.path.join(dirpath, fn))
    return sorted(csvs)


def trace_label(csv_path: str, roots: list[str]) -> str:
    """Derive a per-trace (per-rank) label from the csv path.

    Prefers the ".msprof" trace directory name (one per rank in vllm-ascend's
    torch profiler output), falls back to the first path level below a root.
    """
    abspath = os.path.abspath(csv_path)
    for part in abspath.split(os.sep):
        if ".msprof" in part:
            return part
    for root in roots:
        root = os.path.abspath(root)
        if abspath.startswith(root + os.sep):
            rel = os.path.relpath(abspath, root)
            return rel.split(os.sep)[0]
    return os.path.basename(os.path.dirname(abspath))


def aggregate(csvs: list[str], roots: list[str]):
    """Aggregate raw (sum, count) globally and per trace directory."""
    global_stats: dict[str, list] = {kw: [0.0, 0] for kw in KERNEL_ORDER}
    per_trace: dict[str, dict[str, list]] = defaultdict(lambda: {kw: [0.0, 0] for kw in KERNEL_ORDER})
    for csv_path in csvs:
        label = trace_label(csv_path, roots)
        for kw, (total, count) in parse_csv(csv_path).items():
            global_stats[kw][0] += total
            global_stats[kw][1] += count
            per_trace[label][kw][0] += total
            per_trace[label][kw][1] += count
    return global_stats, per_trace


def group_stats(stats: dict[str, list], phase: str) -> tuple[float, int]:
    total = sum(stats[kw][0] for kw in KERNEL_ORDER if ZBAL_MOE_KERNELS[kw] == phase)
    count = sum(stats[kw][1] for kw in KERNEL_ORDER if ZBAL_MOE_KERNELS[kw] == phase)
    return [total, count]


def write_table(rows, title):
    """Write an aligned markdown table to stdout (style of zbal's perf_analyze.py)."""
    headers = rows[0]
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]

    def fmt_row(row):
        parts = [row[0].ljust(widths[0]) if i == 0 else row[i].rjust(widths[i]) for i in range(len(headers))]
        return "| " + " | ".join(parts) + " |"

    sep = "|" + "|".join(" " + "-" * w + " " for w in widths) + "|"

    print(f"### {title}")
    print()
    print(fmt_row(headers))
    print(sep)
    for row in rows[1:]:
        print(fmt_row(row))
    print()


def run_analyse(roots: list[str]) -> None:
    """Stage 1: run torch_npu offline analysis on every trace directory.

    vllm-ascend's torch profiler leaves one raw msprof trace dir per rank
    (e.g. `<worker_name>.msprof/PROF_*`). `torch_npu.profiler.profiler.analyse`
    with export_type="text" generates kernel_details.csv (and other summary
    csv files) from the raw data, and only scans PROF_* dirs one level below
    its input path, so each trace dir must be analysed individually.
    """
    try:
        from torch_npu.profiler.profiler import analyse
    except ImportError as e:
        sys.exit(
            f"Cannot import torch_npu.profiler.profiler.analyse: {e}\n"
            "Run this script in an environment with torch_npu installed, or pass "
            "--no-analyse when kernel_details.csv files already exist."
        )
    for root in roots:
        if not os.path.isdir(root):
            print(f"[analyse] skip non-directory: {root}")
            continue
        subdirs = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        # When the root itself is a single trace dir (no sub dirs), analyse it
        # directly; otherwise analyse every trace dir below it one by one.
        targets = subdirs or [root]
        for target in targets:
            print(f"[analyse] analysing {target} ...")
            try:
                analyse(target, export_type="text")
                print(f"[analyse] done: {target}")
            except Exception as e:
                # A dir without valid PROF_* data should not abort the rest.
                print(f"[analyse] failed: {target}: {e}", file=sys.stderr)


def report(roots: list[str]):
    csvs = find_kernel_csvs(roots)
    if not csvs:
        print(f"No kernel_details.csv found under: {', '.join(roots)}")
        print("Ensure profiling was collected with vllm-ascend's torch profiler (--profile) first.")
        return

    global_stats, per_trace = aggregate(csvs, roots)

    print("=== ZBAL MoE dispatch/combine kernel performance ===")
    print(f"Roots: {', '.join(roots)}")
    print(f"kernel_details.csv files: {len(csvs)}")
    print()

    # Per-kernel table with group subtotals. "Mean(us/launch)" is the mean
    # duration per kernel launch, aggregated across all ranks (sum/count).
    rows = [["Kernel", "Count", "Total(ms)", "Mean(us/launch)", "Share(%)"]]
    grand_total_us = group_stats(global_stats, "dispatch")[0] + group_stats(global_stats, "combine")[0]

    for phase, phase_title in (("dispatch", "dispatch total"), ("combine", "combine total")):
        for kw in KERNEL_ORDER:
            if ZBAL_MOE_KERNELS[kw] != phase:
                continue
            total_us, count = global_stats[kw]
            if count == 0:
                continue
            share = 100.0 * total_us / grand_total_us if grand_total_us else 0.0
            rows.append(
                [kw, str(count), f"{total_us / 1000.0:.2f}", f"{total_us / count:.2f}", f"{share:.1f}"]
            )
        p_total, p_count = group_stats(global_stats, phase)
        if p_count:
            share = 100.0 * p_total / grand_total_us if grand_total_us else 0.0
            rows.append(
                [f"**{phase_title}**", str(p_count), f"{p_total / 1000.0:.2f}", f"{p_total / p_count:.2f}", f"{share:.1f}"]
            )

    all_count = group_stats(global_stats, "dispatch")[1] + group_stats(global_stats, "combine")[1]
    if all_count:
        rows.append(
            [
                "**dispatch+combine total**",
                str(all_count),
                f"{grand_total_us / 1000.0:.2f}",
                f"{grand_total_us / all_count:.2f}",
                "100.0",
            ]
        )
    write_table(rows, "All ranks (aggregated)")

    if len(per_trace) > 1:
        rows = [["Trace (rank)", "dispatch cnt", "dispatch mean(us)", "combine cnt", "combine mean(us)"]]
        for label in sorted(per_trace):
            stats = per_trace[label]
            d_total, d_count = group_stats(stats, "dispatch")
            c_total, c_count = group_stats(stats, "combine")
            d_mean = f"{d_total / d_count:.2f}" if d_count else "-"
            c_mean = f"{c_total / c_count:.2f}" if c_count else "-"
            rows.append([label, str(d_count), d_mean, str(c_count), c_mean])
        write_table(rows, "Per trace directory")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ZBAL MoE dispatch/combine kernel performance from msprof kernel_details.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dirs", nargs="+", help="profiling output dir(s) containing msprof traces")
    parser.add_argument(
        "--no-analyse",
        action="store_true",
        help="skip the torch_npu.profiler analyse stage (use when kernel_details.csv already exists "
        "or torch_npu is unavailable, e.g. on a non-NPU host)",
    )
    args = parser.parse_args()
    if not args.no_analyse:
        run_analyse(args.dirs)
    report(args.dirs)


if __name__ == "__main__":
    main()
