# Stage-1 HPO results (4 screening images, t0 = 1.0)

`hpo_<model>/results.csv` and `results_per_image.csv` are copies of `outputs/hpo_<model>/`
produced by `configs/experiments_hpo_full_<model>.yaml`.  They let Stage 2 be regenerated
without the gitignored run directories:

    python scripts/hpo_stage2.py --run results/stage1/hpo_jit --run results/stage1/hpo_sit \
        --run results/stage1/hpo_pmf --run results/stage1/hpo_imf --out configs/experiments_hpo_stage2.yaml

Rows with `status != ok` are ignored by every downstream script.
