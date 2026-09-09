# Stage-1 HPO result snapshots

Copies of `outputs/<run>/results.csv` and `results_per_image.csv` for the Stage-1 screening
sweeps (`configs/experiments_hpo_full.yaml`, `configs/experiments_hpo_full_imf.yaml`; 4 images
per configuration, t0 = 1.0).  They exist so that a machine without `outputs/` (which is
gitignored and ~100 GPU-hours to regenerate) can run `hpo_stage2.py` directly:

    python hpo_stage2.py --run results/stage1/hpo_full --run results/stage1/hpo_full_imf \
        --out configs/experiments_hpo_stage2.yaml

`hpo_stage2.py` reads only these CSVs.  Rows with `status != ok` are ignored by it; a snapshot
taken before the redo passes finished therefore has fewer configurations in some cells (see
the "cells without data" line it prints).  Refresh with

    python results/stage1/snapshot.py

and not with `cp`: the run's writer rewrites `results.csv` whole after every job, so a plain
copy from a live run can catch it truncated (it did once: 1775 of 4952 rows).  The script
re-reads until two consecutive reads agree.

Snapshot state is recorded in `SNAPSHOT.txt`.
