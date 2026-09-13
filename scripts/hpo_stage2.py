#!/usr/bin/env python
"""Stage 2 of the per-cell hyperparameter search: who survives the 4-image screen.

    python scripts/hpo_stage2.py --run outputs/hpo_jit [--run outputs/hpo_pmf ...] \
        --out configs/experiments_hpo_stage2.yaml --images 8 --top 3

For every cell (model, problem, method) of a Stage-1 sweep this script

  1. ranks every configuration by mean LPIPS on the screening images;
  2. keeps the top `--top`, adds the manuscript's own configuration when the
     cell has one and it did not already survive, so the paper's choice is always
     re-evaluated at the larger image count;
  3. checks the winner against the grid EDGES: a winner sitting on the smallest or largest
     value of any swept axis is not a local optimum, it is an unfinished search, so one
     more configuration is added with that axis pushed one step further out;
  4. writes a Stage-2 configuration that runs all of those on `--images` images, one
     experiment per (problem, method, candidate rank), batch sizes carried over from Stage 1.

It also prints, per cell, the winner, the paper's rank and LPIPS gap where applicable, and
every boundary hit -- the table that answers "is the manuscript's setting optimal".
Cells with no finished jobs yet are reported and skipped, so it can be run on a partial
sweep to preview the outcome.
"""
import argparse
import collections
import csv
import math
from pathlib import Path

PROBLEMS = ["denoising", "deblur", "super_resolution", "random_inpaint", "box_inpaint"]
EXP_NAME = {"denoising": "denoising", "deblur": "deblurring",
            "super_resolution": "super_resolution", "random_inpaint": "random_inpainting",
            "box_inpaint": "box_inpainting"}
DEGRADATION = {"denoising": "{sigma: 0.20}",
               "deblur": "{sigma: 0.05, blur_sigma: 1.0, kernel_size: 7, padding: reflect}",
               "super_resolution": "{sigma: 0.05, factor: 2}",
               "random_inpaint": "{sigma: 0.01, missing_fraction: 0.70}",
               "box_inpaint": "{sigma: 0.05, box: 40}"}
METHODS = ["sdedit", "pnp", "dflow", "mpc_rhc", "mpc_delta_t", "rhso"]

# The knobs each method sweeps (CSV column names), and which are log-like (extend by the
# grid ratio) versus linear (extend by the grid step) when a winner sits on an edge.
KNOBS = {
    "sdedit":      ["steps"],
    "pnp":         ["num_pnp_steps", "gamma0", "alpha"],
    "dflow":       ["num_opt_steps", "lr"],
    "mpc_rhc":     ["K", "lam", "lr"],
    "mpc_delta_t": ["num_mpc_steps", "n_ctrl", "lam", "lr"],
    "rhso":        ["num_rhso_steps", "num_opt_steps", "lr", "mu"],
}
# counts and dimensions double rather than step: an extension of n_ctrl = 20 by the grid's
# linear step would propose 0, which is not a control dimension
LOG_KNOBS = {"gamma0", "lam", "lr", "num_pnp_steps", "num_opt_steps", "steps",
             "n_ctrl", "num_mpc_steps", "num_rhso_steps"}
INTEGER = {"steps", "num_pnp_steps", "num_opt_steps", "K", "num_mpc_steps", "n_ctrl", "num_rhso_steps"}
# hard limits a knob cannot be pushed past
BOUNDS = {"alpha": (0.0, 1.0), "K": (1, 4), "mu": (0.0, None), "steps": (1, None),
          "n_ctrl": (10, None), "num_mpc_steps": (2, None), "num_rhso_steps": (1, None),
          "num_pnp_steps": (5, None)}
# the YAML key each CSV column maps to (identical here, listed for clarity)
YAML_KEY = {k: k for m in KNOBS for k in KNOBS[m]}
# fixed fields every candidate carries, per method and model family
FIXED = {
    "sdedit":      {"jit": "solver: heun", "sit": "solver: heun", "pmf": "", "imf": ""},
    "pnp":         {m: "" for m in ("jit", "sit", "pmf", "imf")},
    "dflow":       {"jit": "steps: 1, solver: euler", "sit": "steps: 1, solver: euler", "pmf": "steps: 1", "imf": "steps: 1"},
    "mpc_rhc":     {m: "num_mpc_steps: 4, n_ctrl: 40" for m in ("jit", "sit", "pmf", "imf")},
    "mpc_delta_t": {m: "" for m in ("jit", "sit", "pmf", "imf")},
    "rhso":        {"jit": "solver: heun", "sit": "solver: heun", "pmf": "", "imf": ""},
}
# Stage-1 batch sizes (screening ran 4 images; 8 images just means two batches)
BATCH = {("jit", "sdedit"): 4, ("jit", "pnp"): 4, ("jit", "dflow"): 4, ("jit", "mpc_rhc"): 4,
         ("jit", "mpc_delta_t"): 4, ("jit", "rhso"): 4,
         ("sit", "sdedit"): 4, ("sit", "pnp"): 4, ("sit", "dflow"): 4, ("sit", "mpc_rhc"): 4,
         ("sit", "mpc_delta_t"): 4, ("sit", "rhso"): 2,
         ("pmf", "sdedit"): 4, ("pmf", "pnp"): 4, ("pmf", "dflow"): 4, ("pmf", "mpc_rhc"): 2,
         ("pmf", "mpc_delta_t"): 4, ("pmf", "rhso"): 4,
         ("imf", "sdedit"): 2, ("imf", "pnp"): 2, ("imf", "dflow"): 2, ("imf", "mpc_rhc"): 2,
         ("imf", "mpc_delta_t"): 2, ("imf", "rhso"): 2}
# standard-flow RHSO memory grows with N; the budget rows use smaller batches
BATCH_RHSO_LONG = {"jit": 2, "sit": 1, "pmf": 4, "imf": 2}

# The manuscript's hyperparameter tables, per problem in PROBLEMS order.  Only JiT and pMF are tabulated.
PAPER = {
    ("pmf", "pnp"): [dict(num_pnp_steps=50, gamma0=8e5, alpha=.75), dict(num_pnp_steps=100, gamma0=1.2e6, alpha=.50),
                     dict(num_pnp_steps=20, gamma0=1e5, alpha=.25), dict(num_pnp_steps=20, gamma0=1e5, alpha=.25),
                     dict(num_pnp_steps=100, gamma0=4e5, alpha=.25)],
    ("jit", "pnp"): [dict(num_pnp_steps=50, gamma0=2e5, alpha=.50), dict(num_pnp_steps=100, gamma0=1.2e6, alpha=.50),
                     dict(num_pnp_steps=20, gamma0=1e5, alpha=.25), dict(num_pnp_steps=20, gamma0=2e5, alpha=.25),
                     dict(num_pnp_steps=100, gamma0=4e5, alpha=.25)],
    ("pmf", "dflow"): [dict(num_opt_steps=320, lr=.03)] * 5,
    ("jit", "dflow"): [dict(num_opt_steps=320, lr=.03)] * 5,
    ("pmf", "mpc_rhc"): [dict(K=2, lam=15, lr=.10), dict(K=1, lam=60, lr=.05), dict(K=3, lam=240, lr=.05),
                         dict(K=3, lam=180, lr=.05), dict(K=1, lam=30, lr=.05)],
    ("jit", "mpc_rhc"): [dict(K=2, lam=15, lr=.05), dict(K=1, lam=60, lr=.10), dict(K=3, lam=240, lr=.05),
                         dict(K=3, lam=180, lr=.05), dict(K=1, lam=30, lr=.05)],
    ("pmf", "mpc_delta_t"): [dict(num_mpc_steps=8, n_ctrl=40, lam=30, lr=.10), dict(num_mpc_steps=8, n_ctrl=40, lam=540, lr=.10),
                             dict(num_mpc_steps=4, n_ctrl=20, lam=360, lr=.10), dict(num_mpc_steps=4, n_ctrl=20, lam=540, lr=.15),
                             dict(num_mpc_steps=8, n_ctrl=40, lam=180, lr=.10)],
    ("jit", "mpc_delta_t"): [dict(num_mpc_steps=8, n_ctrl=40, lam=30, lr=.10), dict(num_mpc_steps=8, n_ctrl=40, lam=540, lr=.10),
                             dict(num_mpc_steps=4, n_ctrl=20, lam=360, lr=.10), dict(num_mpc_steps=4, n_ctrl=20, lam=540, lr=.10),
                             dict(num_mpc_steps=8, n_ctrl=40, lam=180, lr=.10)],
    ("pmf", "rhso"): [dict(num_rhso_steps=4, num_opt_steps=40, lr=.03, mu=0.0), dict(num_rhso_steps=4, num_opt_steps=40, lr=.01, mu=0.0),
                      dict(num_rhso_steps=4, num_opt_steps=40, lr=.01, mu=0.0), dict(num_rhso_steps=4, num_opt_steps=40, lr=.01, mu=0.0),
                      dict(num_rhso_steps=4, num_opt_steps=40, lr=.02, mu=0.0)],
    ("jit", "rhso"): [dict(num_rhso_steps=4, num_opt_steps=40, lr=.01, mu=0.0)] * 5,
}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def key_of(row, method):
    return tuple(fnum(row.get(k)) for k in KNOBS[method])


def same(a, b):
    return all((x is None and y is None) or (x is not None and y is not None and
               abs(x - y) <= 1e-6 * max(1.0, abs(y))) for x, y in zip(a, b))


def fmt(v, knob):
    if v is None:
        return "null"
    if knob in INTEGER:
        return str(int(round(v)))
    return ("%g" % v) if abs(v) < 1e4 else ("%.1f" % v)


def load(runs):
    """cell -> {config key -> dict(lpips, n_images, rows)}; grids per (model, method)."""
    cells = collections.defaultdict(dict)
    grids = collections.defaultdict(lambda: collections.defaultdict(set))
    for run in runs:
        rows = list(csv.DictReader(open(Path(run) / "results.csv")))
        for r in rows:
            if r["problem"] not in PROBLEMS or r["method"] not in METHODS:
                continue
            if r.get("status", "ok") != "ok":
                continue
            lp = fnum(r["lpips"])
            if lp is None:
                continue
            k = key_of(r, r["method"])
            cell = (r["model"], r["problem"], r["method"])
            cells[cell][k] = dict(lpips=lp, psnr=fnum(r["psnr"]), n=int(r["num_images"]), row=r)
            for knob, v in zip(KNOBS[r["method"]], k):
                if v is not None:
                    grids[(r["model"], r["method"])][knob].add(v)
    return cells, grids


def extend(value, knob, grid, direction):
    """One grid step beyond `value` in `direction` (+1 / -1), or None if not allowed."""
    g = sorted(grid)
    if len(g) < 2:
        return None
    if knob in LOG_KNOBS:
        ratio = (g[-1] / g[0]) ** (1.0 / (len(g) - 1)) if g[0] > 0 else 2.0
        ratio = max(ratio, 1.5)
        new = value * ratio if direction > 0 else value / ratio
    else:
        step = (g[-1] - g[0]) / (len(g) - 1)
        new = value + direction * step
    lo, hi = BOUNDS.get(knob, (None, None))
    if lo is not None and new < lo - 1e-9:
        return None
    if hi is not None and new > hi + 1e-9:
        return None
    if knob in INTEGER:
        new = max(1, int(round(new)))
        if any(abs(new - x) < 0.5 for x in g):
            return None
        return new
    # two significant figures: 0.545136 is not a learning rate anyone would write down
    return float("%.2g" % new)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="Stage-1 run directory (repeatable)")
    p.add_argument("--out", default="configs/experiments_hpo_stage2.yaml")
    p.add_argument("--images", type=int, default=8)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--pool", default="cache/data/imagenet_val_100")
    p.add_argument("--only-model", default=None,
                   help="emit a Stage-2 config for this model only (one file per GPU on multi-GPU machines)")
    args = p.parse_args()

    cells, grids = load(args.run)
    if args.only_model:
        cells = {c: v for c, v in cells.items() if c[0] == args.only_model}
    models = sorted({c[0] for c in cells})
    print("cells with data: %d   models: %s\n" % (len(cells), models))

    # ---- per-cell selection ---------------------------------------------------------------
    candidates = collections.defaultdict(list)          # (problem, method) -> [(model, key, why)]
    print("%-4s %-17s %-12s %6s  %-44s %8s %s" % ("mdl", "problem", "method", "n_cfg", "winner", "LPIPS", "paper rank / gap / edge hits"))
    for model in models:
        for problem in PROBLEMS:
            for method in METHODS:
                cell = (model, problem, method)
                if cell not in cells:
                    continue
                table = cells[cell]
                ranked = sorted(table.items(), key=lambda kv: kv[1]["lpips"])
                knobs = KNOBS[method]
                win_key, win = ranked[0]
                chosen = [(k, "top%d" % (i + 1)) for i, (k, _) in enumerate(ranked[:args.top])]
                note = []
                # the manuscript's configuration, if this cell has one
                pk = PAPER.get((model, method))
                if pk:
                    pv = pk[PROBLEMS.index(problem)]
                    pkey = tuple(float(pv[k]) if k in pv else None for k in knobs)
                    # fill knobs the paper leaves implicit from the winner (e.g. mu = 0 is explicit)
                    pos = [i for i, (k, _) in enumerate(ranked) if same(k, pkey)]
                    if pos:
                        rank = pos[0] + 1
                        gap = ranked[pos[0]][1]["lpips"] - win["lpips"]
                        note.append("paper #%d (+%.4f)" % (rank, gap))
                        if rank > args.top:
                            chosen.append((ranked[pos[0]][0], "paper"))
                    else:
                        note.append("paper cfg NOT in sweep")
                # edge hits on the winner
                for i, knob in enumerate(knobs):
                    v = win_key[i]
                    grid = grids[(model, method)][knob]
                    if v is None or len(grid) < 2:
                        continue
                    g = sorted(grid)
                    for direction, edge in ((-1, g[0]), (+1, g[-1])):
                        if abs(v - edge) <= 1e-9 * max(1, abs(edge)):
                            new = extend(v, knob, grid, direction)
                            if new is not None:
                                k2 = list(win_key); k2[i] = new
                                chosen.append((tuple(k2), "edge:%s%s" % (knob, "+" if direction > 0 else "-")))
                                note.append("edge %s=%s -> try %s" % (knob, fmt(v, knob), fmt(new, knob)))
                            else:
                                note.append("edge %s=%s (hard bound)" % (knob, fmt(v, knob)))
                desc = "  ".join("%s=%s" % (k, fmt(v, k)) for k, v in zip(knobs, win_key))
                print("%-4s %-17s %-12s %6d  %-44s %8.4f %s" % (model, problem, method, len(table), desc, win["lpips"], "; ".join(note)))
                for k, why in chosen:
                    candidates[(problem, method)].append((model, k, why))
    missing = [(m, pr, me) for m in models for pr in PROBLEMS for me in METHODS if (m, pr, me) not in cells]
    if missing:
        print("\n%d cell(s) without data yet (skipped): %s%s" % (len(missing), missing[:6], " ..." if len(missing) > 6 else ""))

    # ---- Stage-2 YAML -----------------------------------------------------------------------
    lines = ['''# =====================================================================================
# STAGE 2 -- confirm the Stage-1 survivors on %d images
# =====================================================================================
#
#   python run.py --config %s --run-id hpo_stage2 --no-figures
#
# Generated by scripts/hpo_stage2.py from %s.  Per cell: the top %d of the 4-image screen, the
# manuscript's configuration where the cell has one, and one step beyond the grid edge on
# every axis where the winner sat on the edge.  `why` in each experiment name records which.
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
  max_atomic_jobs: 4000
  resume: true
  replicate: 0
  gpu_memory_profiling: true
  nvml_sample_interval: 0.02

data:
  source: local_folder
  local_folder: "%s"
  image_size: 256

models:
  jit:
    variant: JiT-B/16
  pmf: {}
  sit: {}
  imf: {}

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
''' % (args.images, args.out, ", ".join(args.run), args.top, args.pool)]
    n_jobs = 0
    for problem in PROBLEMS:
        for method in METHODS:
            cands = candidates.get((problem, method), [])
            if not cands:
                continue
            per_model = collections.defaultdict(list)
            for model, k, why in cands:
                if not any(same(k, k2) for k2, _ in per_model[model]):
                    per_model[model].append((k, why))
            depth = max(len(v) for v in per_model.values())
            for i in range(depth):
                lines.append("\n  %s__%s__c%d:" % (EXP_NAME[problem], method, i + 1))
                lines.append("    enabled: true")
                lines.append("    problem: %s" % problem)
                lines.append("    num_images: %d" % args.images)
                lines.append("    degradation: %s" % DEGRADATION[problem])
                lines.append("    models:")
                for model in ("jit", "sit", "pmf", "imf"):
                    if model not in per_model or i >= len(per_model[model]):
                        continue
                    k, why = per_model[model][i]
                    knobs = KNOBS[method]
                    fields = ["t0: 1.0"]
                    if FIXED[method][model]:
                        fields.append(FIXED[method][model])
                    fields += ["%s: %s" % (YAML_KEY[kn], fmt(v, kn)) for kn, v in zip(knobs, k) if v is not None]
                    bs = BATCH[(model, method)]
                    if method == "rhso" and k[0] is not None and k[0] > 4:
                        bs = BATCH_RHSO_LONG[model]
                    lines.append("      %s:" % model)
                    lines.append("        batch_size: %d" % bs)
                    lines.append("        methods:")
                    lines.append("          %-12s {%s}   # %s" % (method + ":", ", ".join(fields), why))
                    n_jobs += 1
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\nwrote %s: %d jobs at %d images" % (args.out, n_jobs, args.images))


if __name__ == "__main__":
    main()
