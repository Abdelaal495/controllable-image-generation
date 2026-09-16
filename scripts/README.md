# scripts/

Everything here reads a finished run under `outputs/` (or the CSVs committed under
`results/`) and writes a configuration, a table or a figure. None of them trains or
reconstructs anything; `run.py` is the only entry point that touches a GPU.

## Data

| script | in | out |
|---|---|---|
| `build_local_imagenet_pool.py` | Hugging Face | `cache/data/<pool>/`, the image folder `data.source: local_folder` reads. `--frozen-manifest benchmarks/imagenet100_c42_i43/manifest.csv` rebuilds the paper's 100-image benchmark from an ungated mirror |
| `extract_imagenet100_from_tar.py` | a local ImageNet tar | the same pool layout, without the mirror |

## Two-stage hyperparameter search

Stage-1 grids are committed (`configs/experiments_hpo_stage1.yaml`); Stage-2 configurations
are not, because they are derived from Stage-1 results and would go stale.

| script | in | out |
|---|---|---|
| `hpo_stage2.py` | Stage-1 `results.csv` | a Stage-2 config: each cell's top three, the manuscript's value, and one step past any grid edge the winner sits on |
| `hpo_stage2_round.py` | Stage-1 + Stage-2 `results.csv` | a follow-up round for cells whose winner is still on an edge; stops when none is |
| `make_final_frozen100_configs.py` | Stage-2 (+ Stage-1 for `--grid-only`) | `configs/experiments_final_frozen100.yaml`, the argmin-LPIPS winner of every cell |
| `make_best_hyperparameters.py` | Stage-1 + Stage-2 | `docs/best_hyperparameters.md`, including the rank of the manuscript's value in each cell |

## Reports

| script | in | out |
|---|---|---|
| `make_final_tables.py` | `outputs/final_*/` | `results/final/tables.md` |
| `make_results_tex.py` | `outputs/final_*/` | `results/final/results.tex` (booktabs) |
| `paired_tests.py` | `results_per_image.csv` | paired per-image t and Wilcoxon tests between two methods of a cell |
| `make_theory_N_configs.py` | the frozen lr/mu table it carries | the two `configs/experiments_theory_N_fixed*.yaml` stage-count sweeps |
| `make_theory_N_tables.py` | `results/theory_N/` | that experiment's `tables.md`, `theory_N.tex` and `theory_N_sweep.pdf` |

## Bundles

`export_recon_bundle_only.py` and `add_corrupted_images_to_bundle.py` package a run's
reconstructions for sharing; `cluster_modules.sh` selects CUDA/cuDNN modules on Alliance
clusters and is sourced by `setup_cluster.sh`.
