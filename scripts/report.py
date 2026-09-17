#!/usr/bin/env python
"""Turn finished benchmark runs into the reported tables.

    python scripts/report.py --run outputs/final_jit --run outputs/final_pmf \
        --run outputs/final_sit --run outputs/final_imf --out results/final

Writes, into `--out`:
  tables.md                     per cell: LPIPS / PSNR / SSIM / missing-region PSNR /
                                s-per-image, plus a cross-model LPIPS summary
  results.tex                   the same numbers as booktabs tables, one per model
  paired_<metric>_vs_<ref>.md   paired per-image tests against a reference method

Every method of a cell shares the images, measurement, mask and epsilon, so the per-image
differences are paired and much more sensitive than comparing means.  `--paired` selects the
comparisons and repeats.  Runtime columns are only comparable within a run: they depend on
what else shared the GPU.
"""
import argparse
import collections
import csv
import datetime
import math
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

PROBLEMS = ["denoising", "deblur", "super_resolution", "random_inpaint", "box_inpaint"]
METHODS = ["sdedit", "pnp", "dflow", "mpc_rhc", "mpc_delta_t", "rhso"]
MT = {"sdedit": "SDEdit", "pnp": "PnP-Flow", "dflow": "D-Flow", "mpc_rhc": "MPC-RHC",
      "mpc_delta_t": "MPC-Delta_t", "rhso": "RHSO"}
MODEL = {"jit": "JiT-B/16", "pmf": "pMF-L/16", "sit": "SiT-XL/2", "imf": "iMF-B-2"}
PT = {"denoising": "Denoising", "deblur": "Deblurring", "super_resolution": "2x SR",
      "random_inpaint": "Random inpainting", "box_inpaint": "Box inpainting"}
PT_TEX = {"denoising": "Denoising", "deblur": "Deblurring", "super_resolution": "2$\\times$ SR",
          "random_inpaint": "Random inpaint.", "box_inpaint": "Box inpaint."}
MT_TEX = {"sdedit": "SDEdit", "pnp": "PnP-Flow", "dflow": "D-Flow", "mpc_rhc": "MPC-RHC",
          "mpc_delta_t": "MPC-$\\Delta t$", "rhso": "RHSO (ours)"}
MODELS_TEX = [("pmf", "pMF-L/16 (pixel, MeanFlow)"), ("jit", "JiT-B/16 (pixel, standard flow)"),
              ("imf", "iMF-B-2 (latent, MeanFlow)"), ("sit", "SiT-XL/2 (latent, standard flow)")]
HP = {"sdedit": ["steps"], "pnp": ["num_pnp_steps", "gamma0", "alpha"],
      "dflow": ["num_opt_steps", "lr"], "mpc_rhc": ["K", "lam", "lr"],
      "mpc_delta_t": ["num_mpc_steps", "n_ctrl", "lam", "lr"],
      "rhso": ["num_rhso_steps", "num_opt_steps", "lr", "mu"]}
HN = {"steps": "steps", "num_pnp_steps": "N", "gamma0": "\\gamma_0", "alpha": "\\alpha",
      "num_opt_steps": "M", "lr": "\\mathrm{lr}", "K": "K", "lam": "\\lambda",
      "num_mpc_steps": "N", "n_ctrl": "n_\\mathrm{ctrl}", "num_rhso_steps": "N", "mu": "\\mu"}


def f(x, d=4):
    try:
        return ("%%.%df" % d) % float(x)
    except (TypeError, ValueError):
        return "-"


def g(v):
    x = float(v)
    return ("%g" % x) if abs(x) < 1e4 else ("%.0f" % x)


def load(runs):
    """(model, problem, method) -> summary row; job_id -> per-image rows."""
    rows, per = {}, collections.defaultdict(list)
    for run in runs:
        for r in csv.DictReader(open(Path(run) / "results.csv")):
            if r.get("status", "ok") == "ok":
                rows[(r["model"], r["problem"], r["method"])] = r
        path = Path(run) / "results_per_image.csv"
        if path.exists():
            for r in csv.DictReader(open(path)):
                per[r["job_id"]].append(r)
    return rows, per


def lpips_of(per, row):
    return [float(r["lpips"]) for r in per.get(row["job_id"], []) if r.get("lpips")]


def markdown(rows, per, runs, out):
    models = [m for m in MODEL if any(k[0] == m for k in rows)]
    L = ["# Final benchmark on the frozen ImageNet-100 (class seed 42 / image seed 43), "
         "t0 = 1.0, 100 images per cell", "",
         "Generated %s by `scripts/report.py` from %s."
         % (datetime.date.today().isoformat(), ", ".join(runs)), "",
         "Each cell runs the tuned configuration of its (model, problem, method); `+/- se` is "
         "the standard error of LPIPS over the 100 images. Runtimes were measured with one "
         "process per A100-80GB.", ""]
    L += ["## LPIPS summary (lower is better)", "",
          "| model | problem | " + " | ".join(MT[m] for m in METHODS) + " | best |",
          "|---|---|" + "---|" * (len(METHODS) + 1)]
    for m in models:
        for pr in PROBLEMS:
            lp = [float(rows[(m, pr, me)]["lpips"]) if (m, pr, me) in rows
                  and rows[(m, pr, me)].get("lpips") else None for me in METHODS]
            best = min((x for x in lp[1:] if x is not None), default=None)
            cells = ["-" if x is None else
                     ("**%.4f**" % x if me != "sdedit" and x == best else "%.4f" % x)
                     for me, x in zip(METHODS, lp)]
            L.append("| %s | %s | %s | %s |"
                     % (MODEL[m], PT[pr], " | ".join(cells),
                        MT[METHODS[1 + lp[1:].index(best)]] if best is not None else "-"))
    L.append("")
    for m in models:
        L += ["## %s" % MODEL[m], ""]
        for pr in PROBLEMS:
            d = next((rows[(m, pr, me)] for me in METHODS if (m, pr, me) in rows), None)
            if d is None:
                continue
            L += ["### %s" % PT[pr], "",
                  "| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |",
                  "|---|---|---|---|---|---|---|---|",
                  "| degraded input | %s | - | %s | %s | - | - | |"
                  % (f(d["degraded_lpips"]), f(d["degraded_psnr"], 2), f(d["degraded_ssim"]))]
            for me in METHODS:
                r = rows.get((m, pr, me))
                if r is None:
                    L.append("| %s | - | - | - | - | - | - | (missing) |" % MT[me])
                    continue
                values = lpips_of(per, r)
                se = (statistics.stdev(values) / math.sqrt(len(values))
                      if len(values) > 1 else float("nan"))
                L.append("| %s | %s | %.4f | %s | %s | %s | %s | %s |"
                         % (MT[me], f(r["lpips"]), se, f(r["psnr"], 2), f(r["ssim"]),
                            f(r["missing_psnr"], 2) if pr.endswith("inpaint") else "-",
                            f(r["runtime_per_image"], 1),
                            " ".join("%s=%s" % (k, r[k]) for k in HP[me]
                                     if r.get(k) not in (None, ""))))
            L.append("")
    (out / "tables.md").write_text("\n".join(L) + "\n")
    print("wrote %s (%d models, %d cells)" % (out / "tables.md", len(models), len(rows)))


def latex(rows, runs, out):
    L = ["% Generated by scripts/report.py from " + ", ".join(runs),
         "% Frozen ImageNet-100 (class seed 42 / image seed 43), t0 = 1.0, beta = 1, seed 42, "
         "100 images per cell.",
         "% Requires \\usepackage{booktabs}.", ""]
    for m, title in MODELS_TEX:
        if not any(k[0] == m for k in rows):
            continue
        L += ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.5pt}",
              "\\caption{%s on the frozen ImageNet-100 at $t_0 = 1$: LPIPS$\\downarrow$ / "
              "PSNR$\\uparrow$ / SSIM$\\uparrow$ over 100 images. Best guided method per column "
              "in bold.}" % title,
              "\\label{tab:final_%s}" % m,
              "\\begin{tabular}{l" + "ccc" * len(PROBLEMS) + "}", "\\toprule",
              " & " + " & ".join("\\multicolumn{3}{c}{%s}" % PT_TEX[pr] for pr in PROBLEMS) + " \\\\",
              " ".join("\\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(PROBLEMS))),
              "Method & " + " & ".join("LPIPS & PSNR & SSIM" for _ in PROBLEMS) + " \\\\", "\\midrule"]
        best = {}
        for pr in PROBLEMS:
            guided = [rows[(m, pr, me)] for me in METHODS[1:] if (m, pr, me) in rows]
            best[(pr, "lpips")] = min(float(r["lpips"]) for r in guided)
            best[(pr, "psnr")] = max(float(r["psnr"]) for r in guided)
            best[(pr, "ssim")] = max(float(r["ssim"]) for r in guided)
        degraded = [rows[(m, pr, "rhso")] for pr in PROBLEMS]
        L.append("Degraded & " + " & ".join(
            "%.3f & %.2f & %.3f" % (float(r["degraded_lpips"]), float(r["degraded_psnr"]),
                                    float(r["degraded_ssim"])) for r in degraded) + " \\\\")
        L.append("\\midrule")
        for me in METHODS:
            cells = []
            for pr in PROBLEMS:
                r = rows.get((m, pr, me))
                if r is None:
                    cells.append("-- & -- & --")
                    continue
                v = [("lpips", float(r["lpips"]), "%.3f"), ("psnr", float(r["psnr"]), "%.2f"),
                     ("ssim", float(r["ssim"]), "%.3f")]
                cells.append(" & ".join(
                    ("\\textbf{%s}" % (fmt % x)) if (me != "sdedit" and abs(x - best[(pr, k)]) < 1e-9)
                    else (fmt % x) for k, x, fmt in v))
            L.append("%s & %s \\\\" % (MT_TEX[me], " & ".join(cells)))
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]

    L += ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
          "\\caption{LPIPS$\\downarrow$ of every method on the frozen ImageNet-100 at $t_0 = 1$, "
          "all four priors. Best guided method per row in bold.}",
          "\\label{tab:final_lpips_summary}",
          "\\begin{tabular}{ll" + "c" * len(METHODS) + "}", "\\toprule",
          "Prior & Problem & " + " & ".join(MT_TEX[me] for me in METHODS) + " \\\\", "\\midrule"]
    for m, title in MODELS_TEX:
        if not any(k[0] == m for k in rows):
            continue
        for i, pr in enumerate(PROBLEMS):
            vals = [float(rows[(m, pr, me)]["lpips"]) if (m, pr, me) in rows else None
                    for me in METHODS]
            b = min(v for v in vals[1:] if v is not None)
            cells = ["--" if v is None else
                     ("\\textbf{%.3f}" % v if (j > 0 and abs(v - b) < 1e-9) else "%.3f" % v)
                     for j, v in enumerate(vals)]
            L.append("%s & %s & %s \\\\" % (title.split(" (")[0] if i == 0 else "",
                                            PT_TEX[pr], " & ".join(cells)))
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}", "\\end{table}", ""]

    L += ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
          "\\caption{Hyperparameters used in every cell of the final benchmark.}",
          "\\label{tab:final_hparams}",
          "\\begin{tabular}{ll" + "l" * len(PROBLEMS) + "}", "\\toprule",
          "Prior & Method & " + " & ".join(PT_TEX[pr] for pr in PROBLEMS) + " \\\\", "\\midrule"]
    for m, title in MODELS_TEX:
        if not any(k[0] == m for k in rows):
            continue
        for i, me in enumerate(METHODS):
            cells = []
            for pr in PROBLEMS:
                r = rows.get((m, pr, me))
                cells.append("--" if r is None else "$" + ",\\ ".join(
                    "%s{=}%s" % (HN[k], g(r[k])) for k in HP[me]
                    if r.get(k) not in (None, "")) + "$")
            L.append("%s & %s & %s \\\\" % (title.split(" (")[0] if i == 0 else "",
                                            MT_TEX[me], " & ".join(cells)))
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}", "\\end{table*}", ""]
    (out / "results.tex").write_text("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (out / "results.tex", len(L)))


def paired(rows, per, metric, ref, out):
    """Per-image differences against `ref` inside each (model, problem) cell."""
    values = collections.defaultdict(dict)          # (model, problem, method) -> {image: value}
    for cell, row in rows.items():
        for r in per.get(row["job_id"], []):
            try:
                values[cell][r["image_id"]] = float(r[metric])
            except (KeyError, ValueError):
                pass
    L = ["# Paired per-image tests (%s), reference = %s" % (metric, ref), "",
         "| model | problem | method | n | mean %s | mean ref | mean diff (method - ref) | "
         "paired t | p (t) | p (Wilcoxon) | ref wins |" % metric,
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, pr, me) in sorted(values, key=lambda k: (k[0], k[1], METHODS.index(k[2])
                                                     if k[2] in METHODS else 9)):
        if me == ref or (m, pr, ref) not in values:
            continue
        a, b = values[(m, pr, me)], values[(m, pr, ref)]
        ids = sorted(set(a) & set(b))
        if len(ids) < 2:
            continue
        x = np.array([a[i] for i in ids])
        y = np.array([b[i] for i in ids])
        t, pt = stats.ttest_rel(x, y)
        try:
            pw = stats.wilcoxon(x, y).pvalue
        except ValueError:
            pw = float("nan")
        better = (y < x) if metric == "lpips" else (y > x)
        L.append("| %s | %s | %s | %d | %.4f | %.4f | %+.4f | %+.2f | %.2e | %.2e | %d/%d |"
                 % (m, pr, me, len(ids), x.mean(), y.mean(), (x - y).mean(), t, pt, pw,
                    int(better.sum()), len(ids)))
    path = out / ("paired_%s_vs_%s.md" % (metric, ref))
    path.write_text("\n".join(L) + "\n")
    print("wrote %s (%d rows)" % (path, len(L) - 4))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="finished run directory (repeatable)")
    p.add_argument("--out", required=True, help="directory the tables are written to")
    p.add_argument("--paired", action="append", default=None, metavar="METRIC:REF",
                   help="a paired comparison (repeatable; default lpips:rhso, psnr:rhso, lpips:sdedit)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, per = load(a.run)
    markdown(rows, per, a.run, out)
    latex(rows, a.run, out)
    for spec in (a.paired or ["lpips:rhso", "psnr:rhso", "lpips:sdedit"]):
        metric, _, ref = spec.partition(":")
        paired(rows, per, metric, ref or "rhso", out)


if __name__ == "__main__":
    main()
