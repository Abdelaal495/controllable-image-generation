#!/usr/bin/env python
"""Turn Stage-2 winners into the final-benchmark configuration.

    python scripts/make_final_frozen100_configs.py --run outputs/stage2_jit [--run outputs/stage2_pmf ...] \
        --grid-only outputs/hpo_jit [--grid-only outputs/hpo_pmf ...] \
        --out configs/experiments_final_frozen100.yaml

Every model named by --models gets a block under each experiment, so one file serves all four
priors and `run.py --models <m>` picks one.  The model of a Stage-2 row is read from the CSV, so
the --run and --grid-only directories may be given in any order.

Per (model, problem, method) the configuration with the lowest mean LPIPS among the Stage-2 rows
(status == ok) is selected; with --grid-only, only candidates present in the Stage-1 grid are
eligible.  Every problem is split into `__core` (sdedit, pnp, dflow, mpc_delta_t), `__rhc`
(mpc_rhc) and `__rhso` experiments so that batch_size, which resolves per (experiment, model),
can follow each memory class; seeds derive from (problem params, image id) only, so the split
leaves measurement, mask and epsilon identical across the three.
"""
import argparse, collections, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hpo_stage2 import PROBLEMS, EXP_NAME, DEGRADATION, METHODS, KNOBS, FIXED, INTEGER, fnum, fmt, key_of, same

GROUP = {"core": ["sdedit", "pnp", "dflow", "mpc_delta_t"], "rhc": ["mpc_rhc"], "rhso": ["rhso"]}
VARIANT = {"jit": "\n    variant: JiT-B/16", "sit": " {}", "pmf": " {}", "imf": " {}"}
# batch_size per (model, memory class), measured on an 80 GB card
BATCH = {"jit": {"core": 20, "rhc": 10, "rhso": 8}, "sit": {"core": 16, "rhc": 10, "rhso": 6},
         "pmf": {"core": 10, "rhc": 5, "rhso": 10}, "imf": {"core": 4, "rhc": 4, "rhso": 4}}


def load_cells(runs, model, grid=None):
    cells = {}
    for run in runs:
        for r in csv.DictReader(open(Path(run) / "results.csv")):
            if r["model"] != model or r.get("status", "ok") != "ok" or r["problem"] not in PROBLEMS:
                continue
            lp = fnum(r["lpips"])
            if lp is None:
                continue
            if grid is not None and not any(same(key_of(r, r["method"]), g) for g in grid[r["method"]]):
                continue                      # --grid-only: grid-edge extensions are excluded
            key = (r["problem"], r["method"])
            if key not in cells or lp < cells[key]["lpips"]:
                cells[key] = dict(lpips=lp, psnr=fnum(r["psnr"]), n=int(r["num_images"]), row=r)
    return cells


def method_line(method, model, row):
    fields = ["t0: 1.0"]
    fixed = FIXED[method][model]
    if fixed:
        fields.append(fixed)
    for k in KNOBS[method]:
        v = fnum(row.get(k))
        if v is not None:
            fields.append("%s: %s" % (k, fmt(v, k)))
    return "{%s}" % ", ".join(fields)


def summarise(runs):
    """`outputs/stage2_jit, outputs/stage2_jit_r2, ...` is unreadable in a header."""
    stems = sorted({Path(r).name.split("_r")[0] for r in runs})
    return "%s (+ follow-up rounds)" % ", ".join(stems) if len(stems) < len(runs) else ", ".join(stems)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="Stage-2 run directory (repeatable)")
    p.add_argument("--models", default="jit,sit,pmf,imf")
    p.add_argument("--out", required=True)
    p.add_argument("--batch", action="append", default=[], metavar="MODEL:core=N,rhc=N,rhso=N",
                   help="override the built-in batch sizes for one model (repeatable)")
    p.add_argument("--pool", default="cache/data/imagenet100_c42_i43_mirror")
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--grid-only", action="append", default=None, metavar="STAGE1_RUN",
                   help="restrict the winner to configurations present in this Stage-1 grid (repeatable); edge extensions are never selected")
    a = p.parse_args()
    models = [m for m in a.models.split(",") if m]
    batch = {m: dict(BATCH[m]) for m in models}
    for spec in a.batch:
        model, _, kvs = spec.partition(":")
        batch[model].update({k: int(v) for k, v in (kv.split("=") for kv in kvs.split(","))})

    grids = {}
    if a.grid_only:
        for model in models:
            grid = collections.defaultdict(list)
            for run in a.grid_only:
                for r in csv.DictReader(open(Path(run) / "results.csv")):
                    if r["model"] == model and r["problem"] in PROBLEMS and r["method"] in METHODS:
                        grid[r["method"]].append(key_of(r, r["method"]))
            if grid:
                grids[model] = grid
    cells = {m: load_cells(a.run, m, grids.get(m)) for m in models}
    missing = [(m, pr, me) for m in models for pr in PROBLEMS for me in METHODS if (pr, me) not in cells[m]]
    if missing:
        raise SystemExit("no Stage-2 winner for cells: %s" % missing)

    out = ["""# =====================================================================================
# FINAL BENCHMARK at t0 = 1.0 on the frozen ImageNet-100 (class seed 42 / image seed 43)
# =====================================================================================
#
#   python scripts/build_local_imagenet_pool.py --frozen-manifest benchmarks/imagenet100_c42_i43/manifest.csv
#   python run.py --config configs/experiments_final_frozen100.yaml --models %s --run-id final_%s --no-figures
#
# All four priors in one file; run one per GPU with --models.  Generated by
# scripts/make_final_frozen100_configs.py%s from %s: every entry is the lowest-LPIPS
# Stage-2 configuration of its cell among the candidates%s, and the trailing comment is that
# Stage-2 score.  Each problem is split into __core / __rhc / __rhso so batch_size can follow
# memory use.
# =====================================================================================

runtime:
  seed: 42
  accelerator: auto
  output_root: outputs
  cache_root: cache
  release_model_after_use: true
  continue_on_experiment_error: true
  progress_bars: false
  save_individual_images: true
  verbose: true
  max_atomic_jobs: 200
  resume: true
  replicate: 0
  gpu_memory_profiling: true
  nvml_sample_interval: 0.02

data:
  # Written by scripts/build_local_imagenet_pool.py; point it at your copy.
  source: local_folder
  local_folder: "%s"
  image_size: 256

models:
%s
defaults:
  control_cost_normalization: sum_squared
  optimizer: adam
  warm_start: false
  grad_clip: null
  record_loss_history: false

metrics:
  lpips: true
  lpips_net: alex

experiments:
""" % (models[0], models[0], " --grid-only" if grids else "", summarise(a.run),
       " inside the Stage-1 grid" if grids else "", a.pool,
       "".join("  %s:%s\n" % (m, VARIANT[m]) for m in models))]

    for pr in PROBLEMS:
        for grp, methods in GROUP.items():
            out.append("\n  %s__%s:" % (EXP_NAME[pr], grp))
            out.append("    enabled: true")
            out.append("    problem: %s" % pr)
            out.append("    num_images: %d" % a.num_images)
            out.append("    degradation: %s" % DEGRADATION[pr])
            out.append("    models:")
            for model in models:
                out.append("      %s:" % model)
                out.append("        batch_size: %d" % batch[model][grp])
                out.append("        methods:")
                for me in methods:
                    c = cells[model][(pr, me)]
                    out.append("          %-12s %s   # Stage-2 LPIPS %.4f PSNR %.2f at %d images"
                               % (me + ":", method_line(me, model, c["row"]), c["lpips"], c["psnr"] or 0.0, c["n"]))
    Path(a.out).write_text("\n".join(out) + "\n")
    print("wrote %s (%d models x %d cells)" % (a.out, len(models), len(cells[models[0]])))


if __name__ == "__main__":
    main()
