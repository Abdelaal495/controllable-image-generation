#!/usr/bin/env python
"""Cross-machine consistency of the Stage-1 winners.

    python scripts/compare_stage1_machines.py --new outputs/hpo_jit [--new ...] --old results/stage1_prev/hpo_full [--old ...] [--model jit]

For every cell present in both sweeps: the new winner, the old winner, whether they are the same configuration,
'adjacent' (differ in exactly one knob by one grid step) or different, and the old winner's rank / LPIPS gap inside
the new sweep (the paired, same-machine number that says whether the disagreement matters).
"""
import argparse, collections, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hpo_stage2 import PROBLEMS, METHODS, KNOBS, fnum, fmt, key_of, same


def load(runs):
    cells = collections.defaultdict(dict); grids = collections.defaultdict(lambda: collections.defaultdict(set))
    for run in runs:
        for r in csv.DictReader(open(Path(run) / "results.csv")):
            if r["problem"] not in PROBLEMS or r["method"] not in METHODS or r.get("status", "ok") != "ok": continue
            lp = fnum(r["lpips"])
            if lp is None: continue
            k = key_of(r, r["method"]); cell = (r["model"], r["problem"], r["method"])
            if k not in cells[cell] or lp < cells[cell][k]: cells[cell][k] = lp
            for knob, v in zip(KNOBS[r["method"]], k):
                if v is not None: grids[(r["model"], r["method"])][knob].add(v)
    return cells, grids


def relation(a, b, method, grid):
    if same(a, b): return "same"
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if not ((x is None and y is None) or (x is not None and y is not None and abs(x - y) <= 1e-6 * max(1, abs(y))))]
    if len(diffs) == 1:
        i = diffs[0]; g = sorted(grid[KNOBS[method][i]])
        if a[i] in g and b[i] in g and abs(g.index(a[i]) - g.index(b[i])) == 1: return "adjacent"
    return "different"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", action="append", required=True); p.add_argument("--old", action="append", required=True)
    p.add_argument("--model", default=None)
    a = p.parse_args()
    new, grids = load(a.new); old, _ = load(a.old)
    counts = collections.Counter()
    print("%-4s %-17s %-12s %-40s %-40s %-9s %s" % ("mdl", "problem", "method", "new winner", "old winner", "relation", "old winner in new sweep"))
    for cell in sorted(c for c in new if c in old and (a.model is None or c[0] == a.model)):
        m, pr, me = cell
        nk, nl = min(new[cell].items(), key=lambda kv: kv[1]); ok_, ol = min(old[cell].items(), key=lambda kv: kv[1])
        rel = relation(nk, ok_, me, grids[(m, me)]); counts[rel] += 1
        ranked = sorted(new[cell].items(), key=lambda kv: kv[1])
        pos = [i for i, (k, _) in enumerate(ranked) if same(k, ok_)]
        info = "#%d/%d (+%.4f)" % (pos[0] + 1, len(ranked), ranked[pos[0]][1] - nl) if pos else "not in new sweep"
        d = lambda k: " ".join("%s=%s" % (kn, fmt(v, kn)) for kn, v in zip(KNOBS[me], k) if v is not None)
        print("%-4s %-17s %-12s %-40s %-40s %-9s %s" % (m, pr, me, d(nk), d(ok_), rel, info))
    print("\nsummary:", dict(counts))


if __name__ == "__main__":
    main()
