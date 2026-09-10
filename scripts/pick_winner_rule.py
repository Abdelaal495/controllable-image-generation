#!/usr/bin/env python
"""Compare two per-cell winner rules on the Stage-2 results.

  argmin      : the lowest mean LPIPS among the Stage-2 candidates (what the brief asks for)
  1-se/cheap  : the CHEAPEST candidate whose LPIPS is not significantly worse than the argmin
                (paired per-image t-test over the 8 Stage-2 images, two-sided alpha)

Prints, per model, the projected cost of the final 100-image benchmark under each rule and the
per-cell LPIPS the cheaper rule gives up.
"""
import argparse, csv, glob, collections, sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hpo_stage2 as H

p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--alpha", type=float, default=0.05)
p.add_argument("--glob", default=None); a = p.parse_args()
pat = a.glob or f"outputs/stage2_{a.model}*"
rows = collections.defaultdict(dict)        # (problem, method, key) -> row
per = collections.defaultdict(dict)         # job_id -> {image: lpips}
for f in glob.glob(pat + "/results.csv"):
    for r in csv.DictReader(open(f)):
        if r.get("status", "ok") != "ok" or r["problem"] not in H.PROBLEMS or r["model"] != a.model: continue
        rows[(r["problem"], r["method"])][H.key_of(r, r["method"])] = r
for f in glob.glob(pat + "/results_per_image.csv"):
    for r in csv.DictReader(open(f)):
        try: per[r["job_id"]][r["image_id"]] = float(r["lpips"])
        except (KeyError, ValueError): pass

tot_a = tot_c = 0.0; give_up = []
print("%-17s %-12s %-38s %7s %6s   %-38s %7s %6s %s" % ("problem","method","argmin","LPIPS","hours","cheapest not worse","LPIPS","hours","p"))
for (prob, meth), table in sorted(rows.items()):
    best_k, best = min(table.items(), key=lambda kv: float(kv[1]["lpips"]))
    hb = float(best["runtime_per_image"]) * 100 / 3600
    cand = []
    for k, r in table.items():
        h = float(r["runtime_per_image"]) * 100 / 3600
        if h >= hb: continue
        ids = sorted(set(per.get(r["job_id"], {})) & set(per.get(best["job_id"], {})))
        if len(ids) < 3: continue
        x = np.array([per[r["job_id"]][i] for i in ids]); y = np.array([per[best["job_id"]][i] for i in ids])
        pv = stats.ttest_rel(x, y).pvalue
        if pv > a.alpha: cand.append((h, k, r, pv))
    tot_a += hb
    if cand:
        h, k, r, pv = min(cand)
        tot_c += h; give_up.append(float(r["lpips"]) - float(best["lpips"]))
        d = lambda kk: " ".join("%s=%s" % (n, H.fmt(v, n)) for n, v in zip(H.KNOBS[meth], kk) if v is not None)
        print("%-17s %-12s %-38s %7.4f %6.1f   %-38s %7.4f %6.1f %.2f" % (prob, meth, d(best_k), float(best["lpips"]), hb, d(k), float(r["lpips"]), h, pv))
    else:
        tot_c += hb
print("\n%s: argmin rule %.0f GPU-hours; cheapest-not-significantly-worse %.0f GPU-hours (%.0f%% less); median LPIPS given up %.4f over %d cell(s)"
      % (a.model, tot_a, tot_c, 100*(1-tot_c/tot_a), float(np.median(give_up)) if give_up else 0.0, len(give_up)))
