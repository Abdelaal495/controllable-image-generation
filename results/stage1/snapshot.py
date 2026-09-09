#!/usr/bin/env python
"""Copy Stage-1 result CSVs into results/stage1/ safely.

    python results/stage1/snapshot.py [run-id ...]      # default: hpo_full hpo_full_imf

run.py's writer rewrites results.csv from scratch after every job, so a plain `cp` from a
live run can capture a truncated file (it did: 1775 of 4952 rows).  Read until two
consecutive reads agree, then write.
"""
import csv, collections, datetime, hashlib, sys, time
from pathlib import Path

RUNS = tuple(sys.argv[1:]) or ("hpo_full", "hpo_full_imf")   # run-ids as arguments
root = Path(__file__).resolve().parents[2]

def stable_read(path: Path, tries: int = 20) -> bytes:
    prev = None
    for _ in range(tries):
        data = path.read_bytes()
        if prev is not None and hashlib.sha256(data).digest() == hashlib.sha256(prev).digest():
            return data
        prev = data
        time.sleep(1.0)
    raise SystemExit("%s kept changing for %d s; try again" % (path, tries))

lines = ["taken: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")]
for run in RUNS:
    src = root / "outputs" / run
    if not src.exists():
        continue
    dst = root / "results" / "stage1" / run
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("results.csv", "results_per_image.csv"):
        (dst / name).write_bytes(stable_read(src / name))
    rows = list(csv.DictReader(open(dst / "results.csv")))
    c = collections.Counter(r["status"] for r in rows)
    lines.append("%s: %d rows, ok %d, failed %d" % (run, len(rows), c.get("ok", 0), sum(v for k, v in c.items() if k != "ok")))
(root / "results" / "stage1" / "SNAPSHOT.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
