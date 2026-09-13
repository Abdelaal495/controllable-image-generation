#!/usr/bin/env python
"""Final-benchmark tables: 4 models x 5 problems x 6 methods on the frozen ImageNet-100.

    python scripts/make_final_tables.py --run outputs/final_jit [--run ...] --out results/final/tables.md

Per model and problem: LPIPS / PSNR / SSIM (mean over the 100 images), the standard error of LPIPS, the missing-region
PSNR for the inpainting problems, and the per-image runtime (one process per GPU).  The degraded observation is shown as the first row.
"""
import argparse, collections, csv, datetime, math, statistics
from pathlib import Path

PROBLEMS = ["denoising", "deblur", "super_resolution", "random_inpaint", "box_inpaint"]
METHODS = ["sdedit", "pnp", "dflow", "mpc_rhc", "mpc_delta_t", "rhso"]
MT = {"sdedit": "SDEdit", "pnp": "PnP-Flow", "dflow": "D-Flow", "mpc_rhc": "MPC-RHC", "mpc_delta_t": "MPC-Delta_t", "rhso": "RHSO"}
MODEL = {"jit": "JiT-B/16", "pmf": "pMF-L/16", "sit": "SiT-XL/2", "imf": "iMF-B-2"}
PT = {"denoising": "Denoising", "deblur": "Deblurring", "super_resolution": "2x SR", "random_inpaint": "Random inpainting", "box_inpaint": "Box inpainting"}


def f(x, d=4):
    try: return ("%%.%df" % d) % float(x)
    except (TypeError, ValueError): return "-"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = {}; per = collections.defaultdict(list); degraded = {}
    for run in a.run:
        for r in csv.DictReader(open(Path(run) / "results.csv")):
            if r.get("status", "ok") != "ok": continue
            rows[(r["model"], r["problem"], r["method"])] = r
            degraded[(r["model"], r["problem"])] = r
        pi = Path(run) / "results_per_image.csv"
        if pi.exists():
            for r in csv.DictReader(open(pi)):
                try: per[r["job_id"]].append(float(r["lpips"]))
                except (KeyError, ValueError): pass
    models = [m for m in MODEL if any(k[0] == m for k in rows)]
    L = ["# Final benchmark on the frozen ImageNet-100 (class seed 42 / image seed 43), t0 = 1.0, 100 images per cell",
         "", "Generated %s by `scripts/make_final_tables.py` from %s." % (datetime.date.today().isoformat(), ", ".join(a.run)), "",
         "Each cell runs the Stage-2 winner of its (model, problem, method); `+/- se` is the standard error of LPIPS over the 100 images. "
         "Runtimes were measured with one process per A100-80GB.", ""]
    # compact cross-model LPIPS table first
    L += ["## LPIPS summary (lower is better)", "", "| model | problem | " + " | ".join(MT[m] for m in METHODS) + " | best |", "|---|---|" + "---|" * (len(METHODS) + 1)]
    for m in models:
        for pr in PROBLEMS:
            vals = [rows.get((m, pr, me)) for me in METHODS]
            lp = [float(v["lpips"]) if v and v.get("lpips") else None for v in vals]
            best = min((x for x in lp[1:] if x is not None), default=None)
            cells = []
            for me, x in zip(METHODS, lp):
                if x is None: cells.append("-")
                else: cells.append("**%.4f**" % x if me != "sdedit" and x == best else "%.4f" % x)
            L.append("| %s | %s | %s | %s |" % (MODEL[m], PT[pr], " | ".join(cells), MT[METHODS[1 + lp[1:].index(best)]] if best is not None else "-"))
    L.append("")
    for m in models:
        L += ["## %s" % MODEL[m], ""]
        for pr in PROBLEMS:
            d = degraded.get((m, pr))
            if d is None: continue
            L += ["### %s" % PT[pr], "", "| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |", "|---|---|---|---|---|---|---|---|",
                  "| degraded input | %s | - | %s | %s | - | - | |" % (f(d["degraded_lpips"]), f(d["degraded_psnr"], 2), f(d["degraded_ssim"]))]
            for me in METHODS:
                r = rows.get((m, pr, me))
                if r is None: L.append("| %s | - | - | - | - | - | - | (missing) |" % MT[me]); continue
                imgs = per.get(r["job_id"], [])
                se = statistics.stdev(imgs) / math.sqrt(len(imgs)) if len(imgs) > 1 else float("nan")
                hp = {"sdedit": ["steps"], "pnp": ["num_pnp_steps", "gamma0", "alpha"], "dflow": ["num_opt_steps", "lr"],
                      "mpc_rhc": ["K", "lam", "lr"], "mpc_delta_t": ["num_mpc_steps", "n_ctrl", "lam", "lr"], "rhso": ["num_rhso_steps", "num_opt_steps", "lr", "mu"]}[me]
                L.append("| %s | %s | %.4f | %s | %s | %s | %s | %s |" % (MT[me], f(r["lpips"]), se, f(r["psnr"], 2), f(r["ssim"]),
                         f(r["missing_psnr"], 2) if pr.endswith("inpaint") else "-", f(r["runtime_per_image"], 1),
                         " ".join("%s=%s" % (k, r[k]) for k in hp if r.get(k) not in (None, ""))))
            L.append("")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text("\n".join(L) + "\n")
    print("wrote %s (%d models, %d cells)" % (a.out, len(models), len(rows)))


if __name__ == "__main__":
    main()
