#!/usr/bin/env python
"""Paired per-image tests between methods inside one (model, problem) cell of a final run.

    python scripts/paired_tests.py --run outputs/final_jit [--run outputs/final_pmf ...] [--metric lpips] [--ref rhso] --out results/final/paired_tests.md

Every method of a cell shares the same images, measurement, mask and epsilon, so the per-image
differences are paired.  For each (model, problem, method != ref) the script reports the mean paired
difference (method - ref), the paired t statistic and two-sided p-value, the Wilcoxon signed-rank
p-value, and how many images the reference wins.  For LPIPS a NEGATIVE mean difference favours the
method; for PSNR/SSIM a positive one does.
"""
import argparse, collections, csv, math
from pathlib import Path
import numpy as np
from scipy import stats

ORDER = ["sdedit", "pnp", "dflow", "mpc_rhc", "mpc_delta_t", "rhso"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True); p.add_argument("--metric", default="lpips")
    p.add_argument("--ref", default="rhso"); p.add_argument("--out", default=None)
    a = p.parse_args()
    per = collections.defaultdict(dict)       # (model, problem, method) -> {image_id: value}
    for run in a.run:
        for r in csv.DictReader(open(Path(run) / "results_per_image.csv")):
            try: v = float(r[a.metric])
            except (KeyError, ValueError): continue
            per[(r["model"], r["problem"], r["method"])][r["image_id"]] = v
    L = ["# Paired per-image tests (%s), reference = %s" % (a.metric, a.ref), "",
         "| model | problem | method | n | mean %s | mean ref | mean diff (method - ref) | paired t | p (t) | p (Wilcoxon) | ref wins |" % a.metric,
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, pr, me) in sorted(per, key=lambda k: (k[0], k[1], ORDER.index(k[2]) if k[2] in ORDER else 9)):
        if me == a.ref or (m, pr, a.ref) not in per: continue
        ref = per[(m, pr, a.ref)]; cur = per[(m, pr, me)]
        ids = sorted(set(ref) & set(cur))
        if len(ids) < 2: continue
        x = np.array([cur[i] for i in ids]); y = np.array([ref[i] for i in ids]); d = x - y
        t, pt = stats.ttest_rel(x, y)
        try: pw = stats.wilcoxon(x, y).pvalue
        except ValueError: pw = float("nan")
        better = (y < x) if a.metric == "lpips" else (y > x)
        L.append("| %s | %s | %s | %d | %.4f | %.4f | %+.4f | %+.2f | %.2e | %.2e | %d/%d |"
                 % (m, pr, me, len(ids), x.mean(), y.mean(), d.mean(), t, pt, pw, int(better.sum()), len(ids)))
    text = "\n".join(L) + "\n"
    if a.out: Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(text); print("wrote", a.out)
    else: print(text)


if __name__ == "__main__":
    main()
