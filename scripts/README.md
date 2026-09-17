# scripts/

Everything here reads a finished run under `outputs/` and writes a configuration, a table or
a figure. None of them reconstructs anything; `run.py` is the only entry point that needs a GPU.

| script | what it does |
|---|---|
| `build_local_imagenet_pool.py` | builds the image folder `data.source: local_folder` reads. `--frozen-manifest benchmarks/imagenet100_c42_i43/manifest.csv` rebuilds the paper's 100-image benchmark from an ungated mirror |
| `hpo.py` | the hyperparameter search: `propose` the next round from a finished one, `finalize` the winners into the benchmark configuration, `report` what won each cell and where the manuscript's value ranked |
| `report.py` | a finished benchmark to `tables.md`, `results.tex` and the paired per-image tests |
| `make_theory_N_configs.py` | the two RHSO stage-count sweeps, from the frozen learning rates it carries |
| `make_theory_N_tables.py` | those sweeps to their tables, LaTeX and figure |

`extract_imagenet100_from_tar.py` builds the pool from a local ImageNet tar instead of the
mirror; `export_recon_bundle_only.py` and `add_corrupted_images_to_bundle.py` package a run's
reconstructions for sharing; `cluster_modules.sh` selects CUDA/cuDNN modules on Alliance
clusters and is sourced by `setup_cluster.sh`.

## The search loop

There are no numbered stages: one round is run, `propose` reads it and writes the next, and
the loop ends when `propose` reports that no cell wants to move.

```bash
python run.py --config configs/experiments_hpo_grid.yaml --models pmf --run-id hpo_pmf --no-figures
python scripts/hpo.py propose --from outputs/hpo_pmf --models pmf --out configs/round2.yaml
python run.py --config configs/round2.yaml --run-id hpo_pmf_r2 --no-figures
python scripts/hpo.py propose --from outputs/hpo_pmf_r2 --grid outputs/hpo_pmf outputs/hpo_pmf_r2 \
    --top 0 --no-paper --models pmf --out configs/round3.yaml        # only cells still on an edge
```

`--from` is what the ranking reads; `--grid` is the union of everything tried so far and
decides what counts as a grid edge. Passing the original grid to `finalize` keeps the winner
inside the pre-registered ranges, which matters because on several axes (D-Flow and PnP step
counts, RHSO `N` and `M`) LPIPS keeps improving with compute far past the grid — a
compute-budget effect rather than a hyperparameter optimum. `report` lists those out-of-grid
candidates separately instead of selecting them.

Only the first round's configuration is committed. Later rounds are a function of results, so
they are generated rather than stored.
