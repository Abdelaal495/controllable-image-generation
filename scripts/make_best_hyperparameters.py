#!/usr/bin/env python
"""docs/best_hyperparameters.md: the Stage-2 winner of every (model, problem, method) cell at t0 = 1.0,
plus the rank/gap of the manuscript's configurations (Tables 4-7) in Stage 1 (4 images) and Stage 2 (8 images).

    python scripts/make_best_hyperparameters.py --stage1 outputs/hpo_jit ... --stage2 outputs/stage2_jit ... \
        --out docs/best_hyperparameters.md
"""
import argparse, collections, csv, datetime, math, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hpo_stage2 import PROBLEMS, METHODS, KNOBS, PAPER, fnum, fmt, key_of, same

MODEL_TITLE = {"jit": "JiT-B/16 (pixel, standard flow, PyTorch)", "pmf": "pMF-L/16 (pixel, MeanFlow, JAX)",
               "sit": "SiT-XL/2 (latent, standard flow, PyTorch)", "imf": "iMF-B-2 (latent, MeanFlow, JAX)"}
METHOD_TITLE = {"sdedit": "SDEdit (baseline)", "pnp": "PnP-Flow", "dflow": "D-Flow", "mpc_rhc": "MPC-RHC",
                "mpc_delta_t": "MPC-Delta_t", "rhso": "RHSO"}
PROBLEM_TITLE = {"denoising": "Denoising (sigma 0.20)", "deblur": "Deblurring (Gaussian 7/1.0, sigma 0.05)",
                 "super_resolution": "2x super-resolution (sigma 0.05)", "random_inpaint": "Random inpainting (70% missing, sigma 0.01)",
                 "box_inpaint": "Box inpainting (40x40, sigma 0.05)"}


def load(runs):
    """cell -> {config key -> row (lowest LPIPS if duplicated)}, plus per-job per-image LPIPS."""
    cells = collections.defaultdict(dict); per_image = collections.defaultdict(list)
    for run in runs:
        for r in csv.DictReader(open(Path(run) / "results.csv")):
            if r["problem"] not in PROBLEMS or r["method"] not in METHODS or r.get("status", "ok") != "ok":
                continue
            lp = fnum(r["lpips"])
            if lp is None: continue
            k = key_of(r, r["method"]); cell = (r["model"], r["problem"], r["method"])
            if k not in cells[cell] or lp < fnum(cells[cell][k]["lpips"]):
                cells[cell][k] = r
        pi = Path(run) / "results_per_image.csv"
        if pi.exists():
            for r in csv.DictReader(open(pi)):
                lp = fnum(r.get("lpips"))
                if lp is not None: per_image[r["job_id"]].append(lp)
    return cells, per_image


def se(values):
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")


def hp(method, row):
    return " ".join("`%s=%s`" % (k, fmt(fnum(row.get(k)), k)) for k in KNOBS[method] if fnum(row.get(k)) is not None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1", action="append", default=[]); p.add_argument("--stage2", action="append", required=True)
    p.add_argument("--out", default="docs/best_hyperparameters.md"); p.add_argument("--models", default="jit,sit,pmf,imf")
    p.add_argument("--grid-only", action="store_true", help="select winners only among configurations present in the Stage-1 grid (the final-benchmark rule)")
    p.add_argument("--appendix", action="append", default=[], help="extra Stage-2 rounds (grid-edge extensions) listed in an appendix, never selected")
    a = p.parse_args()
    s1, _ = load(a.stage1) if a.stage1 else ({}, {})
    s2, pi2 = load(a.stage2)
    if a.grid_only:
        s2 = {cell: {k: r for k, r in tab.items() if cell in s1 and any(same(k, g) for g in s1[cell])} for cell, tab in s2.items()}
        s2 = {c: t for c, t in s2.items() if t}
    ext, _ = load(a.appendix) if a.appendix else ({}, {})
    models = [m for m in a.models.split(",") if any(c[0] == m for c in s2)]
    L = ["# Best hyperparameters per model, strategy and inverse problem at t0 = 1.0",
         "", "Generated %s by `scripts/make_best_hyperparameters.py` from Stage 2 (%s) and Stage 1 (%s)."
         % (datetime.date.today().isoformat(), ", ".join(a.stage2), ", ".join(a.stage1) or "none"),
         "", "Every entry is the lowest-mean-LPIPS configuration of its cell among the Stage-2 candidates "
         "(Stage-1 top three on 4 images, the manuscript's configuration where one exists, and the grid-edge extensions), "
         "re-run on **8 images** of `cache/data/imagenet_val_100`, `t0 = 1.0`, `beta = 1`, seed 42, replicate 0.  "
         "`+/- se` is the standard error of LPIPS over those 8 images.  `edge` marks a winner whose Stage-2 candidate set "
         "still had it on the extreme of an axis (see the Stage-2 generator's log).  For the two inpainting problems read "
         "`missing_psnr` in results.csv alongside these full-image metrics.", "",
         ""]
    if a.grid_only:
        L += ["**Selection rule.** The winner is the lowest-LPIPS configuration among the Stage-2 candidates that lie INSIDE the "
              "pre-registered Stage-1 grid (itself one step wider than the manuscript's ranges on every axis).  Configurations from the "
              "grid-edge extension rounds are excluded from selection; they are reported in the appendix at the end because on several "
              "axes (D-Flow steps, PnP steps, RHSO N/M) LPIPS kept improving with compute far past the grid, which is a compute-budget "
              "effect rather than a hyperparameter optimum.", ""]
    paper_rows = []
    for m in models:
        L += ["---", "", "# %s" % MODEL_TITLE[m], ""]
        for pr in PROBLEMS:
            L += ["## %s" % PROBLEM_TITLE[pr], "", "| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |", "|---|---|---|---|---|---|---|"]
            best = None
            for me in METHODS:
                table = s2.get((m, pr, me), {})
                if not table:
                    L.append("| %s | - | - | - | - | 0 | (no Stage-2 result) |" % METHOD_TITLE[me]); continue
                k, r = min(table.items(), key=lambda kv: fnum(kv[1]["lpips"]))
                lp = fnum(r["lpips"]); imgs = pi2.get(r["job_id"], [])
                if me != "sdedit" and (best is None or lp < best[0]): best = (lp, me)
                L.append("| %s | `%.4f` | `%s` | `%.2f` | `%.4f` | %d | %s |" % (
                    METHOD_TITLE[me], lp, ("%.4f" % se(imgs)) if len(imgs) > 1 else "-", fnum(r["psnr"]) or 0, fnum(r["ssim"]) or 0, len(table), hp(me, r)))
                pk = PAPER.get((m, me))
                if pk:
                    pv = pk[PROBLEMS.index(pr)]; pkey = tuple(float(pv[x]) if x in pv else None for x in KNOBS[me])
                    def rank_in(tbl):
                        ranked = sorted(tbl.items(), key=lambda kv: fnum(kv[1]["lpips"]))
                        for i, (kk, rr) in enumerate(ranked):
                            if same(kk, pkey): return i + 1, fnum(rr["lpips"]) - fnum(ranked[0][1]["lpips"]), fnum(rr["lpips"]), len(ranked)
                        return None
                    r1 = rank_in(s1.get((m, pr, me), {})) if s1 else None
                    r2 = rank_in(table)
                    paper_rows.append((m, pr, me, " ".join("%s=%s" % (x, fmt(float(pv[x]), x)) for x in KNOBS[me] if x in pv), r1, r2, hp(me, r), lp))
            if best: L.append("\n**best non-baseline:** %s (LPIPS %.4f)" % (METHOD_TITLE[best[1]], best[0]))
            L.append("")
    L += ["---", "", "# Are the manuscript's hyperparameters (Tables 4-7) optimal?", "",
          "Rank of the manuscript's configuration inside each cell, and its LPIPS gap to the cell's winner.  Stage 1 ranks among "
          "all grid configurations on 4 images; Stage 2 ranks among the surviving candidates on 8 images (the manuscript's "
          "configuration is always one of them).  A gap of 0.0000 means the manuscript's choice is the winner.", "",
          "| model | problem | method | manuscript config | Stage-1 rank / gap (4 img) | Stage-2 rank / gap (8 img) | manuscript LPIPS (8 img) | our winner | winner LPIPS |", "|---|---|---|---|---|---|---|---|---|"]
    for m, pr, me, cfg, r1, r2, win, wl in paper_rows:
        f1 = "#%d/%d (+%.4f)" % (r1[0], r1[3], r1[1]) if r1 else "not in sweep"
        f2 = "#%d/%d (+%.4f)" % (r2[0], r2[3], r2[1]) if r2 else "not run"
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | `%.4f` |" % (m, pr, me, cfg, f1, f2, ("`%.4f`" % r2[2]) if r2 else "-", win, wl))
    if ext:
        L += ["", "---", "", "# Appendix: grid-edge extension rounds (NOT used for the final benchmark)", "",
              "Lowest LPIPS reached in each cell when the edge of the grid was pushed repeatedly (8 images), next to the grid-only winner above.", "",
              "| model | problem | method | grid-only winner LPIPS | extended best LPIPS | extended configuration |", "|---|---|---|---|---|---|"]
        for m in models:
            for pr in PROBLEMS:
                for me in METHODS:
                    if (m, pr, me) not in ext or (m, pr, me) not in s2: continue
                    ek, er = min(ext[(m, pr, me)].items(), key=lambda kv: fnum(kv[1]["lpips"]))
                    gk, gr = min(s2[(m, pr, me)].items(), key=lambda kv: fnum(kv[1]["lpips"]))
                    if fnum(er["lpips"]) >= fnum(gr["lpips"]) - 1e-9: continue
                    L.append("| %s | %s | %s | `%.4f` | `%.4f` | %s |" % (m, pr, me, fnum(gr["lpips"]), fnum(er["lpips"]), hp(me, er)))
    Path(a.out).write_text("\n".join(L) + "\n")
    print("wrote %s: %d models, %d manuscript cells" % (a.out, len(models), len(paper_rows)))


if __name__ == "__main__":
    main()
