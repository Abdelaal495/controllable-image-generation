#!/usr/bin/env python
"""Cell-by-cell comparison of the frozen-100 final results against the manuscript's Table 1 (pMF) / Table 8 (JiT).

    python scripts/compare_to_paper.py --paper paper_tables.csv --run outputs/final_pmf --run outputs/final_jit [--out results/final/paper_comparison.md]

`paper_tables.csv` columns: model,problem,method,lpips,psnr,ssim  (model in jit/pmf/sit/imf; problem in
denoising/deblur/super_resolution/random_inpaint/box_inpaint; method in sdedit/pnp/dflow/mpc_rhc/mpc_delta_t/rhso;
use method=degraded for the "Degraded" row; leave a metric empty if the paper does not report it).
Same 100 images and operators, so the deltas are directly interpretable; the degraded row should agree within
0.1 dB and is printed first as the sanity check.
"""
import argparse, csv
from pathlib import Path

PROBLEMS = ["denoising", "deblur", "super_resolution", "random_inpaint", "box_inpaint"]
METHODS = ["degraded", "sdedit", "pnp", "dflow", "mpc_rhc", "mpc_delta_t", "rhso"]


def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper", required=True); p.add_argument("--run", action="append", required=True); p.add_argument("--out", default=None)
    a = p.parse_args()
    paper = {(r["model"], r["problem"], r["method"]): r for r in csv.DictReader(open(a.paper))}
    ours = {}
    for run in a.run:
        for r in csv.DictReader(open(Path(run) / "results.csv")):
            if r.get("status", "ok") != "ok": continue
            ours[(r["model"], r["problem"], r["method"])] = {"lpips": f(r["lpips"]), "psnr": f(r["psnr"]), "ssim": f(r["ssim"])}
            ours[(r["model"], r["problem"], "degraded")] = {"lpips": f(r["degraded_lpips"]), "psnr": f(r["degraded_psnr"]), "ssim": f(r["degraded_ssim"])}
    L = ["| model | problem | method | LPIPS ours | LPIPS paper | d | PSNR ours | PSNR paper | d | SSIM ours | SSIM paper | d |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("pmf", "jit", "sit", "imf"):
        for pr in PROBLEMS:
            for me in METHODS:
                if (m, pr, me) not in paper or (m, pr, me) not in ours: continue
                o, q = ours[(m, pr, me)], paper[(m, pr, me)]; cells = []
                for k in ("lpips", "psnr", "ssim"):
                    pv = f(q.get(k)); ov = o[k]
                    cells += ["%.4f" % ov if ov is not None else "-", "%.4f" % pv if pv is not None else "-",
                              ("%+.4f" % (ov - pv)) if (pv is not None and ov is not None) else "-"]
                L.append("| %s | %s | %s | %s |" % (m, pr, me, " | ".join(cells)))
    text = "\n".join(L) + "\n"
    if a.out: Path(a.out).write_text(text); print("wrote", a.out, "(%d rows)" % (len(L) - 2))
    else: print(text)


if __name__ == "__main__":
    main()
