#!/usr/bin/env python3
# ACC Data Tracer
# Yacine - Barcelona Supercomputing Center (BSC)

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MPI_TAG_RE = re.compile(r"^\[(\d+),(\d+)\]<[^>]+>:\s*")
SRUN_LABEL_RE = re.compile(r"^(\d+):\s*")


def parse_args():
    p = argparse.ArgumentParser(
        prog="acc-data-trace.py",
        description="Run a command, add timestamps and ranks to its output, and generate a JSON report.",
    )
    p.add_argument("-out", default="report.json", help="Output JSON file")
    p.add_argument("-log", default="out", help="Raw log file")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run, after --")
    return p.parse_args()


def inject_launcher_tagging(cmd):
    if not cmd:
        return cmd

    exe = os.path.basename(cmd[0])

    if exe in ("mpirun", "mpiexec"):
        if "--tag-output" not in cmd:
            return [cmd[0], "--tag-output", *cmd[1:]]
        return cmd

    if exe == "srun":
        if "--label" not in cmd and "-l" not in cmd:
            return [cmd[0], "--label", *cmd[1:]]
        return cmd

    return cmd


def extract_rank_and_strip_prefix(line, serial_default=False):
    rank = "0" if serial_default else "?"

    m = MPI_TAG_RE.match(line)
    if m:
        return m.group(2), line[m.end():]

    m = SRUN_LABEL_RE.match(line)
    if m:
        return m.group(1), line[m.end():]

    return rank, line


def main():
    args = parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        print(
            "Usage: acc-data-trace.py -out report.json -log out -- <your usual command>",
            file=sys.stderr,
        )
        print(
            "Examples:",
            file=sys.stderr,
        )
        print(
            "  acc-data-trace.py -out report.json -- mpirun -np 4 --bind-to none ./alya case",
            file=sys.stderr,
        )
        print(
            "  acc-data-trace.py -out report.json -- srun -n 4 ./alya case",
            file=sys.stderr,
        )
        print(
            "  acc-data-trace.py -out report.json -- ./app",
            file=sys.stderr,
        )
        sys.exit(2)

    script_dir = Path(__file__).resolve().parent
    parser_script = script_dir / "acc_pipeline_prepare.py"

    if not parser_script.exists():
        print(f"Error: parser script not found: {parser_script}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["NV_ACC_NOTIFY"] = "18"

    run_cmd = inject_launcher_tagging(cmd)
    serial_default = os.path.basename(run_cmd[0]) not in ("mpirun", "mpiexec", "srun")

    t0 = time.perf_counter()

    with open(args.log, "w", buffering=1) as flog:
        proc = subprocess.Popen(
            run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        try:
            for line in proc.stdout:
                dt = time.perf_counter() - t0
                rank, clean_line = extract_rank_and_strip_prefix(line, serial_default=serial_default)
                flog.write(f"[{dt:.6f}][rank={rank}] {clean_line}")
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise

        ret = proc.wait()

    if ret != 0:
        print(f"Command failed with exit code {ret}", file=sys.stderr)
        print("Continuing anyway: parsing the partial log.", file=sys.stderr)

    parser_ret = subprocess.call([sys.executable, str(parser_script), args.log, args.out])
    if parser_ret != 0:
        print(f"Parser failed with exit code {parser_ret}", file=sys.stderr)
        sys.exit(parser_ret)

    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
