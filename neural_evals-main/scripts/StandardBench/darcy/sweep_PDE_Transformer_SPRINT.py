#!/usr/bin/env python3
"""
Grid-search sweep for PDE_Transformer_SPRINT on the Darcy benchmark.

Usage
-----
  # print all commands without running them
  python sweep_PDE_Transformer_SPRINT.py --dry-run

  # run sequentially on a single GPU
  python sweep_PDE_Transformer_SPRINT.py --gpu 0

  # worker pool: 4 GPUs each run one job at a time, all in parallel
  python sweep_PDE_Transformer_SPRINT.py --gpus 0,1,2,3

Customisation
-------------
  Edit FIXED_ARGS  – options that stay the same for every run.
  Edit SWEEP_ARGS  – options to sweep; each key maps to a list of values.
                     Every combination of the listed values will be run.
  Edit NAME_KEYS   – which sweep keys appear in the auto-generated run name
                     (and in what order).  Others are still swept but not
                     reflected in the name.
"""

import argparse
import itertools
import os
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue

# ---------------------------------------------------------------------------
# Fixed arguments (same for every run)
# ---------------------------------------------------------------------------
FIXED_ARGS: dict = {
    "data_path": "./data/darcy",
    "loader": "darcy",
    "geotype": "structured_2D",
    "task": "steady",
    "normalize": 1,
    "derivloss": 1,
    "downsamplex": 5,
    "downsampley": 5,
    "space_dim": 2,
    "fun_dim": 1,
    "out_dim": 1,
    "model": "PDE_Transformer_SPRINT",
    "n_hidden": 96,
    "mlp_ratio": 4,
    "pdet_window_size": 8,
    "pdet_max_hidden": 512,
    "pdet_patch_size": 1,
    "unified_pos": 1,
    "ref": 8,
    "batch_size": 4,   # maps to --batch-size (see flag_name() below)
    "epochs": 500,
    "eval": 0,
    "vis_cbar_min": -0.0003,
    "vis_cbar_max": 0.0003,
}

# Boolean flags (store_true in run.py) – include the key here when you want the
# flag to be present in a run.  To sweep presence/absence, add the key to
# SWEEP_ARGS with values [True, False].
FIXED_FLAGS: list[str] = []

# ---------------------------------------------------------------------------
# Sweep arguments (all combinations will be run)
# ---------------------------------------------------------------------------
SWEEP_ARGS: dict[str, list] = {
    "lr": [5.5e-4, 1e-3],
    "pdet_depth": ["2,4,2", "1,3,1"],
    "use_ema_clip": [0, 1],
    "n_heads": [4, 6, 8],
    # "pdet_use_gated_mlp": [False, True],   # bool → flag present/absent
    "pdet_use_upsample_act": [False, True],
    # "pdet_sprint_drop_mode": ["random", "l2"],
    "pdet_output_act": ["gelu", None],
    # "pdet_sprint_fusion_type": ["linear", "gated"],
    # "n_hidden": [64, 96, 128],
    "max_grad_norm": [None, 1.0],
    # "amp": [0, 1],
    "scheduler": ["OneCycleLR", "CosineAnnealingLR"],
}

# ---------------------------------------------------------------------------
# Run-name configuration
# ---------------------------------------------------------------------------
# Keys whose values appear in the auto-generated --save_name, in this order.
NAME_KEYS: list[str] = [
    "lr",
    "pdet_depth",
    "use_ema_clip",
    "n_heads",
    # "pdet_use_gated_mlp",
    "pdet_use_upsample_act",
    # "pdet_sprint_drop_mode",
    "pdet_output_act",
    "max_grad_norm",
    "scheduler"
]

# Prefix prepended to every run name.
NAME_PREFIX: str = "darcy_PDE_Transformer_SPRINT_sweep"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def flag_name(key: str) -> str:
    """Convert a Python dict key to its CLI flag name.

    All argparse flags in run.py use underscores except --batch-size.
    """
    _REMAP = {
        "batch_size": "batch-size",
    }
    return "--" + _REMAP.get(key, key)


def value_tag(key: str, value) -> str:
    """Short human-readable representation of a sweep value for the run name."""
    if value is None:
        return "default"
    if isinstance(value, bool):
        return key.split("_")[-1] if value else f"no-{key.split('_')[-1]}"
    if isinstance(value, float):
        # e.g. 5.5e-4 → "5.5e-4"
        return f"{value:.2e}".replace("+0", "").replace("-0", "-").replace("+", "")
    # Replace commas in comma-separated strings (e.g. pdet_depth) with dashes
    return str(value).replace(",", "-")


def build_run_name(combo: dict) -> str:
    parts = [NAME_PREFIX]
    for k in NAME_KEYS:
        if k in combo:
            parts.append(f"{k}_{value_tag(k, combo[k])}")
    return "_".join(parts)


def build_command(combo: dict, gpu: str, run_name: str, script_root: Path) -> list[str]:
    cmd = [sys.executable, str(script_root / "run.py"), "--gpu", gpu]

    # Fixed scalar args
    for k, v in FIXED_ARGS.items():
        cmd += [flag_name(k), str(v)]

    # Fixed boolean flags
    for flag in FIXED_FLAGS:
        cmd.append(f"--{flag}")

    # Sweep args for this combo
    for k, v in combo.items():
        if v is None:
            pass  # omit → argparse default applies
        elif isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
            # False → omit flag entirely
        else:
            cmd += [flag_name(k), str(v)]

    cmd += ["--save_name", run_name]
    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cli = argparse.ArgumentParser(description="Grid sweep for PDE_Transformer_SPRINT")
    cli.add_argument("--gpu", type=str, default="0", help="GPU index for sequential runs")
    cli.add_argument("--gpus", type=str, default=None,
                     help="Comma-separated GPU indices; runs one job per GPU concurrently (worker pool)")
    cli.add_argument("--dry-run", action="store_true",
                     help="Print commands without executing them")
    cli.add_argument("--skip-existing", action="store_true",
                     help="Skip a run if its results directory already exists under ./results/")
    args = cli.parse_args()

    gpu_list = args.gpus.split(",") if args.gpus else [args.gpu]

    # Root of the project (three levels up from this script)
    script_root = Path(__file__).resolve().parents[3]

    # Build all (key, value-list) pairs in a stable order
    sweep_keys = list(SWEEP_ARGS.keys())
    sweep_values = [SWEEP_ARGS[k] for k in sweep_keys]
    combos = [dict(zip(sweep_keys, vals)) for vals in itertools.product(*sweep_values)]

    n_workers = len(gpu_list)
    print(f"Total combinations : {len(combos)}")
    print(f"Workers (GPUs)     : {n_workers} ({', '.join(gpu_list)})")
    print(f"Script root        : {script_root}")
    print()

    # Build the job queue
    job_queue: Queue = Queue()
    for i, combo in enumerate(combos):
        run_name = build_run_name(combo)
        if args.skip_existing:
            results_dir = script_root / "results" / run_name
            if results_dir.exists():
                print(f"[skip] {run_name}  (results dir exists)")
                continue
        job_queue.put((i + 1, len(combos), combo, run_name))

    if args.dry_run:
        while not job_queue.empty():
            idx, total, combo, run_name = job_queue.get()
            cmd = build_command(combo, gpu_list[0], run_name, script_root)
            print(f"[{idx}/{total}] {run_name}")
            print(f"  {' '.join(cmd)}")
            print()
        return

    print_lock = threading.Lock()
    failed_runs: list[str] = []

    def worker(gpu: str) -> None:
        while True:
            try:
                idx, total, combo, run_name = job_queue.get_nowait()
            except Exception:
                break
            cmd = build_command(combo, gpu, run_name, script_root)
            with print_lock:
                print(f"[{idx}/{total}] START  gpu={gpu}  {run_name}")
            ret = subprocess.run(cmd, cwd=str(script_root), env=os.environ.copy())
            with print_lock:
                status = "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"
                print(f"[{idx}/{total}] {status} gpu={gpu}  {run_name}")
            if ret.returncode != 0:
                failed_runs.append(run_name)
            job_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpu_list]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print()
    if failed_runs:
        print(f"[ERROR] {len(failed_runs)} run(s) failed:", file=sys.stderr)
        for name in failed_runs:
            print(f"  {name}", file=sys.stderr)
        sys.exit(1)
    print("Sweep complete.")


if __name__ == "__main__":
    main()
