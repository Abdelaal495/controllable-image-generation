#!/usr/bin/env python
"""Hyperparameter search for every (model, problem, method) cell.

One round at a time: run a configuration, `propose` the next one from its results, repeat
until no cell wants to move, then `finalize` the winners into the benchmark configuration.

    # round 1 -- the pre-registered grid
    python run.py --config configs/experiments_hpo_grid.yaml --models pmf --run-id hpo_pmf --no-figures

    # round 2 -- the top three of each cell, the manuscript's value, and one step past any
    #            axis whose winner sat on the edge of the grid, re-run on more images
    python scripts/hpo.py propose --from outputs/hpo_pmf --models pmf --images 8 --out configs/round2.yaml
    python run.py --config configs/round2.yaml --run-id hpo_pmf_r2 --no-figures

    # round 3+ -- only the cells still sitting on an edge of everything tried so far
    python scripts/hpo.py propose --from outputs/hpo_pmf_r2 --grid outputs/hpo_pmf outputs/hpo_pmf_r2 \
        --top 0 --no-paper --models pmf --images 8 --out configs/round3.yaml
    # `propose` prints "no cell needs another push" and writes nothing when the search is done

    # the benchmark configuration, and the document that justifies every value in it
    python scripts/hpo.py finalize --from outputs/hpo_pmf_r2 --grid outputs/hpo_pmf \
        --models pmf --out configs/experiments_final_frozen100.yaml
    python scripts/hpo.py report --from outputs/hpo_pmf_r2 --screen outputs/hpo_pmf \
        --out docs/best_hyperparameters.md

`--grid` is the union of every value tried so far and decides what counts as a grid edge;
`--from` is what the ranking reads, and defaults `--grid` to itself.  Restricting `finalize`
to a `--grid` keeps the winner inside the pre-registered ranges: on several axes (D-Flow and
PnP step counts, RHSO N and M) LPIPS keeps improving with compute far past the grid, which is
a compute-budget effect rather than a hyperparameter optimum.
"""
import argparse
import collections
import csv
import datetime
import math
import statistics
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
# batch_size per (model, method) for a screening round
BATCH = {("jit", "sdedit"): 4, ("jit", "pnp"): 4, ("jit", "dflow"): 4, ("jit", "mpc_rhc"): 4,
         ("jit", "mpc_delta_t"): 4, ("jit", "rhso"): 4,
         ("sit", "sdedit"): 4, ("sit", "pnp"): 4, ("sit", "dflow"): 4, ("sit", "mpc_rhc"): 4,
         ("sit", "mpc_delta_t"): 4, ("sit", "rhso"): 2,
         ("pmf", "sdedit"): 4, ("pmf", "pnp"): 4, ("pmf", "dflow"): 4, ("pmf", "mpc_rhc"): 2,
         ("pmf", "mpc_delta_t"): 4, ("pmf", "rhso"): 4,
         ("imf", "sdedit"): 2, ("imf", "pnp"): 2, ("imf", "dflow"): 2, ("imf", "mpc_rhc"): 2,
         ("imf", "mpc_delta_t"): 2, ("imf", "rhso"): 2}
# standard-flow RHSO memory grows with N, so long-N candidates use smaller batches
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


def load(runs, per_image=False):
    """cell -> {config key -> dict(lpips, psnr, n, row)}, the grid each (model, method) covers,
    and, optionally, the per-image LPIPS of every job.

    A configuration re-run in several rounds keeps its BEST row, so the result does not depend
    on the order the run directories are given in.
    """
    cells = collections.defaultdict(dict)
    grids = collections.defaultdict(lambda: collections.defaultdict(set))
    images = collections.defaultdict(list)
    for run in runs:
        for r in csv.DictReader(open(Path(run) / "results.csv")):
            if r["problem"] not in PROBLEMS or r["method"] not in METHODS:
                continue
            if r.get("status", "ok") != "ok":
                continue
            lp = fnum(r["lpips"])
            if lp is None:
                continue
            k = key_of(r, r["method"])
            cell = (r["model"], r["problem"], r["method"])
            if k not in cells[cell] or lp < cells[cell][k]["lpips"]:
                cells[cell][k] = dict(lpips=lp, psnr=fnum(r["psnr"]), n=int(r["num_images"]), row=r)
            for knob, v in zip(KNOBS[r["method"]], k):
                if v is not None:
                    grids[(r["model"], r["method"])][knob].add(v)
        if per_image:
            path = Path(run) / "results_per_image.csv"
            if path.exists():
                for r in csv.DictReader(open(path)):
                    lp = fnum(r.get("lpips"))
                    if lp is not None:
                        images[r["job_id"]].append(lp)
    return cells, grids, images


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


def edge_pushes(win_key, method, grid):
    """Every (new key, why) that moves the winner one step off an edge of `grid`."""
    out = []
    for i, knob in enumerate(KNOBS[method]):
        v = win_key[i]
        axis = grid.get(knob, set())
        if v is None or len(axis) < 2:
            continue
        g = sorted(axis)
        for direction, edge in ((-1, g[0]), (+1, g[-1])):
            if abs(v - edge) <= 1e-9 * max(1, abs(edge)):
                new = extend(v, knob, axis, direction)
                if new is None:
                    out.append((None, "%s=%s hard bound" % (knob, fmt(v, knob))))
                else:
                    k2 = list(win_key)
                    k2[i] = new
                    out.append((tuple(k2), "edge:%s%s" % (knob, "+" if direction > 0 else "-")))
    return out


def paper_key(model, method, problem):
    """The manuscript's configuration for a cell, as a knob tuple, or None."""
    table = PAPER.get((model, method))
    if not table:
        return None
    values = table[PROBLEMS.index(problem)]
    return tuple(float(values[k]) if k in values else None for k in KNOBS[method])


def batch_for(model, method, key):
    if method == "rhso" and (key[0] or 0) >= 8:
        return BATCH_RHSO_LONG[model]
    return BATCH[(model, method)]


def method_fields(method, model, key):
    fields = ["t0: 1.0"] + ([FIXED[method][model]] if FIXED[method][model] else [])
    return fields + ["%s: %s" % (YAML_KEY[kn], fmt(v, kn))
                     for kn, v in zip(KNOBS[method], key) if v is not None]


HEADER = """# =====================================================================================
# %s
# =====================================================================================
#
%s#
# %s
# Generated by scripts/hpo.py %s from %s.
# =====================================================================================

runtime:
  seed: 42
  accelerator: auto
  output_root: outputs
  cache_root: cache
  release_model_after_use: true
  continue_on_experiment_error: true
  progress_bars: false
  save_individual_images: %s
  verbose: true
  max_atomic_jobs: %d
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
"""


def model_block(models):
    return "".join("  %s:%s\n" % (m, "\n    variant: JiT-B/16" if m == "jit" else " {}")
                   for m in models)


# =====================================================================================
# propose -- the next round of candidates
# =====================================================================================
def propose(a):
    models = [m for m in a.models.split(",") if m]
    rank, _, _ = load(getattr(a, "from"))
    _, grids, _ = load(a.grid or getattr(a, "from"))
    candidates = collections.defaultdict(list)          # (problem, method) -> [(model, key, why)]
    print("%-4s %-17s %-12s %6s  %-44s %8s %s"
          % ("mdl", "problem", "method", "n_cfg", "winner", "LPIPS", "notes"))
    for model in models:
        for problem in PROBLEMS:
            for method in METHODS:
                table = rank.get((model, problem, method))
                if not table:
                    continue
                ranked = sorted(table.items(), key=lambda kv: kv[1]["lpips"])
                win_key, win = ranked[0]
                chosen = [(k, "top%d" % (i + 1)) for i, (k, _) in enumerate(ranked[:a.top])]
                notes = []
                if a.paper:
                    pkey = paper_key(model, method, problem)
                    if pkey is not None:
                        hit = [i for i, (k, _) in enumerate(ranked) if same(k, pkey)]
                        if hit:
                            notes.append("paper #%d (+%.4f)"
                                         % (hit[0] + 1, ranked[hit[0]][1]["lpips"] - win["lpips"]))
                            if hit[0] + 1 > a.top:
                                chosen.append((ranked[hit[0]][0], "paper"))
                        else:
                            notes.append("paper cfg NOT in sweep")
                for key, why in edge_pushes(win_key, method, grids[(model, method)]):
                    if key is None:
                        notes.append(why)
                    elif not any(same(key, k) for k in table):
                        chosen.append((key, why))
                        notes.append(why)
                desc = "  ".join("%s=%s" % (k, fmt(v, k)) for k, v in zip(KNOBS[method], win_key))
                print("%-4s %-17s %-12s %6d  %-44s %8.4f %s"
                      % (model, problem, method, len(table), desc, win["lpips"], "; ".join(notes)))
                for key, why in chosen:
                    candidates[(problem, method)].append((model, key, why))
    if not candidates:
        print("\nno cell needs another push: every winner has a worse neighbour on both sides "
              "(or a hard bound), so the search is finished")
        return

    lines = [HEADER % (
        "CANDIDATE ROUND at %d images" % a.images,
        "#   python run.py --config %s --models <model> --run-id <id> --no-figures\n" % a.out,
        "Each entry is a candidate this round should settle: the best of its cell so far, the\n"
        "# manuscript's configuration where the cell has one, or one step past a grid edge the\n"
        "# winner sat on.  The trailing comment on each line says which.",
        "propose", ", ".join(getattr(a, "from")), "true", 4000, a.pool, model_block(models))]
    n_jobs = 0
    for problem in PROBLEMS:
        for method in METHODS:
            per_model = collections.defaultdict(list)
            for model, key, why in candidates.get((problem, method), []):
                if not any(same(key, k) for k, _ in per_model[model]):
                    per_model[model].append((key, why))
            if not per_model:
                continue
            for i in range(max(len(v) for v in per_model.values())):
                lines.append("\n  %s__%s__c%d:" % (EXP_NAME[problem], method, i + 1))
                lines.append("    enabled: true")
                lines.append("    problem: %s" % problem)
                lines.append("    num_images: %d" % a.images)
                lines.append("    degradation: %s" % DEGRADATION[problem])
                lines.append("    models:")
                for model in models:
                    if i >= len(per_model.get(model, [])):
                        continue
                    key, why = per_model[model][i]
                    lines.append("      %s:" % model)
                    lines.append("        batch_size: %d" % batch_for(model, method, key))
                    lines.append("        methods:")
                    lines.append("          %-12s {%s}   # %s"
                                 % (method + ":", ", ".join(method_fields(method, model, key)), why))
                    n_jobs += 1
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\nwrote %s: %d jobs at %d images" % (a.out, n_jobs, a.images))


# =====================================================================================
# finalize -- the winner of every cell, as the benchmark configuration
# =====================================================================================
GROUP = {"core": ["sdedit", "pnp", "dflow", "mpc_delta_t"], "rhc": ["mpc_rhc"], "rhso": ["rhso"]}
# batch_size per (model, memory class) at 100 images, measured on an 80 GB card
BATCH_FINAL = {"jit": {"core": 20, "rhc": 10, "rhso": 8}, "sit": {"core": 16, "rhc": 10, "rhso": 6},
               "pmf": {"core": 10, "rhc": 5, "rhso": 10}, "imf": {"core": 4, "rhc": 4, "rhso": 4}}


def restrict(cells, models, grids=None):
    """(model, problem, method) -> the candidate table the winner is chosen from.

    With `grids`, only configurations whose every knob was part of the searched grid survive,
    so a candidate produced by pushing an axis past its edge cannot win.  Rank and gap are
    then reported against the same table, which is what keeps the two comparable.
    """
    tables = {}
    for model in models:
        for problem in PROBLEMS:
            for method in METHODS:
                table = cells.get((model, problem, method), {})
                if grids is not None:
                    axes = grids.get((model, method))
                    table = {k: v for k, v in table.items()
                             if axes and all(kv is None or kv in axes.get(kn, set())
                                             for kn, kv in zip(KNOBS[method], k))}
                if table:
                    tables[(model, problem, method)] = table
    return tables


def winners(tables):
    """(model, problem, method) -> (key, entry, candidates it beat)."""
    return {cell: (lambda kv: (kv[0], kv[1], len(t)))(min(t.items(), key=lambda kv: kv[1]["lpips"]))
            for cell, t in tables.items()}


def finalize(a):
    models = [m for m in a.models.split(",") if m]
    batch = {m: dict(BATCH_FINAL[m]) for m in models}
    for spec in a.batch:
        model, _, kvs = spec.partition(":")
        batch[model].update({k: int(v) for k, v in (kv.split("=") for kv in kvs.split(","))})
    cells, _, _ = load(getattr(a, "from"))
    grids = load(a.grid)[1] if a.grid else None
    best = winners(restrict(cells, models, grids))
    missing = [(m, pr, me) for m in models for pr in PROBLEMS for me in METHODS
               if (m, pr, me) not in best]
    if missing:
        raise SystemExit("no winner for cells: %s" % missing)

    stems = sorted({Path(r).name.split("_r")[0] for r in getattr(a, "from")})
    source = ", ".join(stems) + (" (+ follow-up rounds)" if len(stems) < len(getattr(a, "from")) else "")
    out = [HEADER % (
        "FINAL BENCHMARK at t0 = 1.0 on the frozen ImageNet-100 (class seed 42 / image seed 43)",
        "#   python scripts/build_local_imagenet_pool.py --frozen-manifest "
        "benchmarks/imagenet100_c42_i43/manifest.csv\n"
        "#   python run.py --config %s --models <model> --run-id <id> --no-figures\n" % a.out,
        "All four priors in one file; run one per GPU with --models.  Every entry is the\n"
        "# lowest-LPIPS configuration of its cell%s, and the trailing\n"
        "# comment is that score.  Each problem is split into __core / __rhc / __rhso so that\n"
        "# batch_size, which resolves per (experiment, model), can follow each memory class."
        % (" among the candidates inside the searched grid" if grids else ""),
        "finalize", source, "true", 200, a.pool, model_block(models))]
    for problem in PROBLEMS:
        for group, methods in GROUP.items():
            out.append("\n  %s__%s:" % (EXP_NAME[problem], group))
            out.append("    enabled: true")
            out.append("    problem: %s" % problem)
            out.append("    num_images: %d" % a.num_images)
            out.append("    degradation: %s" % DEGRADATION[problem])
            out.append("    models:")
            for model in models:
                out.append("      %s:" % model)
                out.append("        batch_size: %d" % batch[model][group])
                out.append("        methods:")
                for method in methods:
                    key, cell, _ = best[(model, problem, method)]
                    out.append("          %-12s {%s}   # LPIPS %.4f PSNR %.2f at %d tuning images"
                               % (method + ":", ", ".join(method_fields(method, model, key)),
                                  cell["lpips"], cell["psnr"] or 0.0, cell["n"]))
    Path(a.out).write_text("\n".join(out) + "\n")
    print("wrote %s (%d models x %d cells)" % (a.out, len(models), len(PROBLEMS) * len(METHODS)))


# =====================================================================================
# report -- what won each cell, and where the manuscript's value ranked
# =====================================================================================
MODEL_TITLE = {"jit": "JiT-B/16 (pixel, standard flow, PyTorch)", "pmf": "pMF-L/16 (pixel, MeanFlow, JAX)",
               "sit": "SiT-XL/2 (latent, standard flow, PyTorch)", "imf": "iMF-B-2 (latent, MeanFlow, JAX)"}
METHOD_TITLE = {"sdedit": "SDEdit (baseline)", "pnp": "PnP-Flow", "dflow": "D-Flow", "mpc_rhc": "MPC-RHC",
                "mpc_delta_t": "MPC-Delta_t", "rhso": "RHSO"}
PROBLEM_TITLE = {"denoising": "Denoising (sigma 0.20)", "deblur": "Deblurring (Gaussian 7/1.0, sigma 0.05)",
                 "super_resolution": "2x super-resolution (sigma 0.05)",
                 "random_inpaint": "Random inpainting (70% missing, sigma 0.01)",
                 "box_inpaint": "Box inpainting (40x40, sigma 0.05)"}


def se(values):
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")


def hp(method, key):
    return " ".join("`%s=%s`" % (kn, fmt(v, kn))
                    for kn, v in zip(KNOBS[method], key) if v is not None)


def rank_of(table, key):
    """(rank, LPIPS gap to the winner, LPIPS, cell size) of `key`, or None if it was never run."""
    ranked = sorted(table.items(), key=lambda kv: kv[1]["lpips"])
    for i, (k, v) in enumerate(ranked):
        if same(k, key):
            return i + 1, v["lpips"] - ranked[0][1]["lpips"], v["lpips"], len(ranked)
    return None


def report(a):
    models = [m for m in a.models.split(",") if m]
    cells, _, images = load(getattr(a, "from"), per_image=True)
    screen = load(a.screen)[0] if a.screen else {}
    tables = restrict(cells, models, load(a.grid)[1] if a.grid else None)
    best = winners(tables)
    models = [m for m in models if any(c[0] == m for c in best)]

    L = ["# Best hyperparameters per model, strategy and inverse problem at t0 = 1.0", "",
         "Generated %s by `scripts/hpo.py report` from %s."
         % (datetime.date.today().isoformat(), ", ".join(getattr(a, "from"))), "",
         "Every entry is the lowest-mean-LPIPS configuration of its cell%s, on %s images of the "
         "tuning pool at `t0 = 1.0`, `beta = 1`, seed 42, replicate 0.  `+/- se` is the standard "
         "error of LPIPS over those images.  For the two inpainting problems read `missing_psnr` "
         "in results.csv alongside these full-image metrics."
         % (" among the candidates inside the searched grid" if a.grid else "",
            best[next(iter(best))][1]["n"] if best else "the"), ""]
    paper_rows = []
    for model in models:
        L += ["---", "", "# %s" % MODEL_TITLE[model], ""]
        for problem in PROBLEMS:
            L += ["## %s" % PROBLEM_TITLE[problem], "",
                  "| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |",
                  "|---|---|---|---|---|---|---|"]
            champion = None
            for method in METHODS:
                if (model, problem, method) not in best:
                    L.append("| %s | - | - | - | - | 0 | (no result) |" % METHOD_TITLE[method])
                    continue
                key, cell, n_cand = best[(model, problem, method)]
                row, lp = cell["row"], cell["lpips"]
                per = images.get(row["job_id"], [])
                if method != "sdedit" and (champion is None or lp < champion[0]):
                    champion = (lp, method)
                L.append("| %s | `%.4f` | `%s` | `%.2f` | `%.4f` | %d | %s |"
                         % (METHOD_TITLE[method], lp, ("%.4f" % se(per)) if len(per) > 1 else "-",
                            fnum(row["psnr"]) or 0, fnum(row["ssim"]) or 0, n_cand, hp(method, key)))
                pkey = paper_key(model, method, problem)
                if pkey is not None:
                    paper_rows.append((model, problem, method, hp(method, pkey),
                                       rank_of(screen.get((model, problem, method), {}), pkey) if screen else None,
                                       rank_of(tables.get((model, problem, method), {}), pkey),
                                       hp(method, key), lp))
            if champion:
                L.append("\n**best non-baseline:** %s (LPIPS %.4f)" % (METHOD_TITLE[champion[1]], champion[0]))
            L.append("")
    L += ["---", "", "# Are the manuscript's hyperparameters optimal?", "",
          "Rank of the manuscript's configuration inside each cell, and its LPIPS gap to the "
          "cell's winner.  A gap of 0.0000 means the manuscript's choice is the winner.", "",
          "| model | problem | method | manuscript config | screening rank / gap | final rank / gap | manuscript LPIPS | our winner | winner LPIPS |",
          "|---|---|---|---|---|---|---|---|---|"]
    for model, problem, method, cfg, r_screen, r_final, win, win_lpips in paper_rows:
        f = lambda r: "#%d/%d (+%.4f)" % (r[0], r[3], r[1]) if r else "not in sweep"
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | `%.4f` |"
                 % (model, problem, method, cfg, f(r_screen), f(r_final),
                    ("`%.4f`" % r_final[2]) if r_final else "-", win, win_lpips))
    # Whatever the grid restriction excluded, and what it would have scored.  On several axes
    # (D-Flow and PnP step counts, RHSO N and M) LPIPS keeps falling with compute well past the
    # grid, so these are a compute-budget effect and are reported rather than selected.
    outside = [(m, pr, me, best[(m, pr, me)][1]["lpips"], cand)
               for m in models for pr in PROBLEMS for me in METHODS
               if (m, pr, me) in best and (m, pr, me) in cells
               for cand in [min(cells[(m, pr, me)].items(), key=lambda kv: kv[1]["lpips"])]
               if cand[1]["lpips"] < best[(m, pr, me)][1]["lpips"] - 1e-9]
    if outside:
        L += ["", "---", "", "# Candidates outside the searched grid (reported, never selected)", "",
              "Lowest LPIPS reached in each cell once an axis was pushed past the grid, next to the "
              "winner above.", "",
              "| model | problem | method | winner LPIPS | best outside the grid | configuration |",
              "|---|---|---|---|---|---|"]
        for model, problem, method, win_lpips, (key, cell) in outside:
            L.append("| %s | %s | %s | `%.4f` | `%.4f` | %s |"
                     % (model, problem, method, win_lpips, cell["lpips"], hp(method, key)))
    Path(a.out).write_text("\n".join(L) + "\n")
    print("wrote %s: %d models, %d manuscript cells, %d cells improved outside the grid"
          % (a.out, len(models), len(paper_rows), len(outside)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(q, out_default=None):
        q.add_argument("--from", dest="from", action="append", required=True, metavar="RUN",
                       help="run directory whose results.csv is ranked (repeatable)")
        q.add_argument("--grid", action="append", metavar="RUN",
                       help="run directories that define the searched grid (default: --from)")
        q.add_argument("--models", default="jit,sit,pmf,imf")
        q.add_argument("--out", default=out_default, required=out_default is None)

    q = sub.add_parser("propose", help="write the next round of candidates")
    common(q)
    q.add_argument("--top", type=int, default=3, help="candidates kept per cell (0 = edges only)")
    q.add_argument("--paper", action=argparse.BooleanOptionalAction, default=True,
                   help="also re-run the manuscript's configuration where a cell has one")
    q.add_argument("--images", type=int, default=8)
    q.add_argument("--pool", default="cache/data/imagenet_val_100")
    q.set_defaults(run=propose)

    q = sub.add_parser("finalize", help="write the benchmark configuration from the winners")
    common(q)
    q.add_argument("--batch", action="append", default=[], metavar="MODEL:core=N,rhc=N,rhso=N",
                   help="override the built-in batch sizes for one model (repeatable)")
    q.add_argument("--pool", default="cache/data/imagenet100_c42_i43_mirror")
    q.add_argument("--num-images", type=int, default=100)
    q.set_defaults(run=finalize)

    q = sub.add_parser("report", help="write the tuned-hyperparameter document")
    common(q, out_default="docs/best_hyperparameters.md")
    q.add_argument("--screen", action="append", metavar="RUN",
                   help="earlier, cheaper rounds, ranked separately in the manuscript table")
    q.set_defaults(run=report)

    a = p.parse_args()
    a.run(a)


if __name__ == "__main__":
    main()
