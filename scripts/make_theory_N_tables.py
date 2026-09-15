#!/usr/bin/env python
"""Tables and the figure for the capacity-matched RHSO stage-count sweeps.

    python scripts/make_theory_N_tables.py [--root outputs] [--out results/theory_N]

Reads outputs/theory_<protocol>_<model>/results{,_per_image}.csv for the two protocols
(fixedB: B = N*M = 160; fixedM: M = 40) and the four models, and writes
  <out>/tables.md            five-task means per N, per-task tables, MeanFlow-minus-FM deltas
                             with paired per-image tests (same images and measurements)
  <out>/theory_N_sweep.pdf   five-task mean PSNR and LPIPS against N, one panel per protocol
                             and metric (plus a .png preview)
  <out>/theory_N.tex         the five-task means and per-task tables as booktabs LaTeX
"""
import argparse, collections, csv, datetime, math
from pathlib import Path
import numpy as np

PROTOCOLS = [("fixedB", "Fixed total budget B = N x M = 160"), ("fixedM", "Fixed per-stage budget M = 40")]
PROBLEMS = ["denoising", "deblur", "super_resolution", "random_inpaint", "box_inpaint"]
PT = {"denoising": "Denoising", "deblur": "Deblurring", "super_resolution": "2x SR",
      "random_inpaint": "Random inpainting", "box_inpaint": "Box inpainting"}
MODELS = [("pmf", "pMF-L/16", "pixel", "MeanFlow"), ("jit", "JiT-L/16", "pixel", "Flow Matching"),
          ("imf", "iMF-XL/2", "latent", "MeanFlow"), ("sit", "SiT-XL/2", "latent", "Flow Matching")]
NAME = {m: n for m, n, _, _ in MODELS}
PAIRS = [("pixel", "pmf", "jit"), ("latent", "imf", "sit")]
METRICS = [("psnr", "PSNR", 2), ("ssim", "SSIM", 3), ("lpips", "LPIPS", 3), ("measurement_rmse", "meas. RMSE", 4)]
NOTE = ("Every cell is RHSO with direct terminal planning, t0 = 1, beta = 1, 100 frozen ImageNet-100 images, seed 42. "
        "lr and mu per task are the frozen-100 benchmark winners (docs/best_hyperparameters.md), not the manuscript's values, "
        "and are frozen across N and across model size. Runtimes are not reported: several shards shared each GPU.")


def load(root):
    rows, per = {}, collections.defaultdict(dict)       # (proto, model, problem, N) -> row / {image: {metric: v}}
    for proto, _ in PROTOCOLS:
        for m, _, _, _ in MODELS:
            d = Path(root) / ("theory_%s_%s" % (proto, m))
            if not (d / "results.csv").exists():
                continue
            for r in csv.DictReader(open(d / "results.csv")):
                if r.get("status") != "ok":
                    continue
                rows[(proto, m, r["problem"], int(float(r["num_rhso_steps"])))] = r
            if (d / "results_per_image.csv").exists():
                for r in csv.DictReader(open(d / "results_per_image.csv")):
                    key = (proto, m, r["problem"], int(float(r["num_rhso_steps"])))
                    per[key][r["image_id"]] = {k: float(r[k]) for k, _, _ in METRICS if r.get(k) not in (None, "")}
    return rows, per


def fmt(x, d):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else ("%%.%df" % d) % x


def mean_over_tasks(rows, proto, m, n, metric):
    vals = [float(rows[(proto, m, p, n)][metric]) for p in PROBLEMS if (proto, m, p, n) in rows and rows[(proto, m, p, n)].get(metric)]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def paired(per, proto, a, b, n, metric):
    """Pooled over tasks: per-image (a - b), paired t and Wilcoxon p, and how many images favour a."""
    from scipy import stats
    x, y = [], []
    for p in PROBLEMS:
        ka, kb = (proto, a, p, n), (proto, b, p, n)
        if ka not in per or kb not in per:
            continue
        for img in sorted(set(per[ka]) & set(per[kb])):
            if metric in per[ka][img] and metric in per[kb][img]:
                x.append(per[ka][img][metric]); y.append(per[kb][img][metric])
    if len(x) < 3:
        return None
    x, y = np.array(x), np.array(y); d = x - y
    t, pt = stats.ttest_rel(x, y)
    try:
        pw = stats.wilcoxon(x, y).pvalue
    except ValueError:
        pw = float("nan")
    better = (d < 0) if metric in ("lpips", "measurement_rmse") else (d > 0)
    return dict(n=len(d), mean=float(d.mean()), t=float(t), pt=float(pt), pw=float(pw), wins=int(better.sum()))


def write_tables(rows, per, out):
    L = ["# Capacity-matched RHSO stage-count sweeps", "",
         "Generated %s by `scripts/make_theory_N_tables.py`. %s" % (datetime.date.today().isoformat(), NOTE), "",
         "| model | parameters | space | family |", "|---|---|---|---|"]
    params = {"pmf": "410 M", "jit": "459 M", "imf": "610 M", "sit": "675 M"}
    for m, name, space, fam in MODELS:
        L.append("| %s | %s | %s | %s |" % (name, params[m], space, fam))
    L.append("")
    for proto, title in PROTOCOLS:
        Ns = sorted({k[3] for k in rows if k[0] == proto})
        if not Ns:
            continue
        L += ["## %s" % title, "", "### Five-task means", ""]
        L.append("| N | M | " + " | ".join("%s %s" % (NAME[m], lab) for m, _, _, _ in MODELS for _, lab, _ in METRICS[:3]) + " |")
        L.append("|---|---|" + "---|" * (3 * len(MODELS)))
        for n in Ns:
            mrow = next((rows[k] for k in rows if k[0] == proto and k[3] == n), None)
            cells = []
            for m, _, _, _ in MODELS:
                for metric, _, d in METRICS[:3]:
                    v, cnt = mean_over_tasks(rows, proto, m, n, metric)
                    cells.append(fmt(v, d) + ("" if cnt in (0, 5) else " (%d/5)" % cnt))
            L.append("| %d | %s | %s |" % (n, mrow["num_opt_steps"] if mrow else "-", " | ".join(cells)))
        L.append("")
        # MeanFlow minus Flow Matching, per space, with paired tests pooled over the five tasks
        L += ["### MeanFlow minus Flow Matching (five tasks pooled, paired per image)", "",
              "Positive dPSNR / dSSIM and negative dLPIPS favour the MeanFlow prior. `wins` counts images where the MeanFlow prior is better.", "",
              "| space | N | pairs | dPSNR | p (t) | wins | dLPIPS | p (t) | wins | dSSIM | p (t) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for space, a, b in PAIRS:
            for n in Ns:
                r = {met: paired(per, proto, a, b, n, met) for met in ("psnr", "lpips", "ssim")}
                if not all(r.values()):
                    continue
                L.append("| %s (%s vs %s) | %d | %d | %+.2f | %.1e | %d | %+.4f | %.1e | %d | %+.4f | %.1e |" % (
                    space, NAME[a], NAME[b], n, r["psnr"]["n"], r["psnr"]["mean"], r["psnr"]["pt"], r["psnr"]["wins"],
                    r["lpips"]["mean"], r["lpips"]["pt"], r["lpips"]["wins"], r["ssim"]["mean"], r["ssim"]["pt"]))
        L.append("")
        # per-task
        L += ["### Per task (PSNR / SSIM / LPIPS / measurement RMSE)", ""]
        for p in PROBLEMS:
            L += ["#### %s" % PT[p], "", "| N | " + " | ".join(NAME[m] for m, _, _, _ in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
            for n in Ns:
                cells = []
                for m, _, _, _ in MODELS:
                    r = rows.get((proto, m, p, n))
                    cells.append("-" if r is None else " / ".join(fmt(float(r[k]), d) if r.get(k) else "-" for k, _, d in METRICS))
                L.append("| %d | %s |" % (n, " | ".join(cells)))
            L.append("")
    (out / "tables.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "tables.md")


def write_tex(rows, out):
    L = ["% Generated by scripts/make_theory_N_tables.py. " + NOTE, "% Requires \\usepackage{booktabs,multirow}.", ""]
    TEX = {"pmf": "pMF-L/16", "jit": "JiT-L/16", "imf": "iMF-XL/2", "sit": "SiT-XL/2"}
    TITLE = {"fixedB": "fixed total budget $B = N \\times M = 160$", "fixedM": "fixed per-stage budget $M = 40$"}
    for proto, title in PROTOCOLS:
        Ns = sorted({k[3] for k in rows if k[0] == proto})
        if not Ns:
            continue
        L += ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
              "\\caption{Capacity-matched RHSO stage-count sweep, %s. Five-task means of PSNR$\\uparrow$ / SSIM$\\uparrow$ / LPIPS$\\downarrow$ over 100 images; "
              "best value per metric and column pair in bold.}" % TITLE[proto],
              "\\label{tab:capacity_%s}" % proto, "\\begin{tabular}{rr" + "ccc" * len(MODELS) + "}", "\\toprule",
              "& & " + " & ".join("\\multicolumn{3}{c}{%s}" % TEX[m] for m, _, _, _ in MODELS) + " \\\\",
              " ".join("\\cmidrule(lr){%d-%d}" % (3 + 3 * i, 5 + 3 * i) for i in range(len(MODELS))),
              "$N$ & $M$ & " + " & ".join("PSNR & SSIM & LPIPS" for _ in MODELS) + " \\\\", "\\midrule"]
        vals = {(m, n, met): mean_over_tasks(rows, proto, m, n, met)[0] for m, _, _, _ in MODELS for n in Ns for met, _, _ in METRICS[:3]}
        for n in Ns:
            mrow = next((rows[k] for k in rows if k[0] == proto and k[3] == n), None)
            cells = []
            for space, a, b in PAIRS:
                for m in (a, b):
                    for met, _, d in METRICS[:3]:
                        v = vals[(m, n, met)]; o = vals[(b if m == a else a, n, met)]
                        best = v is not None and o is not None and ((v < o) if met == "lpips" else (v > o))
                        cells.append(("\\textbf{%s}" % fmt(v, d)) if best else fmt(v, d))
            L.append("%d & %s & %s \\\\" % (n, mrow["num_opt_steps"] if mrow else "-", " & ".join(cells)))
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
        # per-task appendix table: rows (task, N), one column per model with PSNR/SSIM/LPIPS
        L += ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
              "\\caption{Per-task results of the capacity-matched sweep, %s. Each cell is PSNR/SSIM/LPIPS.}" % TITLE[proto],
              "\\label{tab:capacity_%s_tasks}" % proto, "\\begin{tabular}{lr" + "c" * len(MODELS) + "}", "\\toprule",
              "Task & $N$ & " + " & ".join(TEX[m] for m, _, _, _ in MODELS) + " \\\\", "\\midrule"]
        for p in PROBLEMS:
            for i, n in enumerate(Ns):
                cells = []
                for m, _, _, _ in MODELS:
                    r = rows.get((proto, m, p, n))
                    cells.append("--" if r is None else "/".join(fmt(float(r[k]), d) for k, _, d in METRICS[:3]))
                L.append("%s & %d & %s \\\\" % (PT[p] if i == 0 else "", n, " & ".join(cells)))
            L.append("\\midrule")
        L[-1] = "\\bottomrule"; L += ["\\end{tabular}", "\\end{table*}", ""]
    (out / "theory_N.tex").write_text("\n".join(L) + "\n")
    print("wrote", out / "theory_N.tex")


def figure(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colour = {"pmf": "#2a78d6", "jit": "#eb6834", "imf": "#1baf7a", "sit": "#eda100"}
    style = {"pmf": "-", "imf": "-", "jit": "--", "sit": "--"}
    marker = {"pmf": "o", "jit": "o", "imf": "s", "sit": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.4), sharex="row")
    for i, (proto, title) in enumerate(PROTOCOLS):
        Ns = sorted({k[3] for k in rows if k[0] == proto})
        for j, (metric, lab, _) in enumerate([("psnr", "PSNR (dB)", 2), ("lpips", "LPIPS", 3)]):
            ax = axes[i, j]
            for m, name, space, fam in MODELS:
                ys = [mean_over_tasks(rows, proto, m, n, metric)[0] for n in Ns]
                pts = [(n, y) for n, y in zip(Ns, ys) if y is not None]
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style[m], marker=marker[m], ms=4, lw=1.6,
                        color=colour[m], label="%s (%s, %s)" % (name, space, fam))
            ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
            ax.grid(True, color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.set_ylabel(lab + (" (higher is better)" if metric == "psnr" else " (lower is better)"), fontsize=8)
            ax.set_title(title, fontsize=8)
            if i == 1:
                ax.set_xlabel("generative intervals N", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("RHSO stage-count sweeps, capacity-matched priors (five-task means)", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(out / ("theory_N_sweep.%s" % ext), dpi=200, bbox_inches="tight")
    print("wrote", out / "theory_N_sweep.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs"); p.add_argument("--out", default="results/theory_N")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, per = load(a.root)
    print("%d cells loaded" % len(rows))
    write_tables(rows, per, out)
    write_tex(rows, out)
    figure(rows, out)


if __name__ == "__main__":
    main()
