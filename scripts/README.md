# scripts/

| script | what it does |
|---|---|
| `make_frozen_selection.py` | draws a frozen benchmark offline as (class, rank) pairs: the 1000-image `benchmarks/imagenet1000_c42_i43` (one validation image per class, tuning-pool rows excluded); reproduces the 100-image plan with `--num-classes 100 --no-exclude` |
| `build_local_imagenet_pool.py` | builds the image folder `data.source: local_folder` reads. `--frozen-manifest benchmarks/imagenet100_c42_i43/manifest.csv` rebuilds the paper's 100-image benchmark from an ungated mirror, `--frozen-manifest benchmarks/imagenet1000_c42_i43/manifest.csv` the 1000-image one; each lands in its own `cache/data/<manifest dir>_mirror` |
| `hpo.py` | the hyperparameter search: `propose` the next round of candidates from a finished one, `finalize` the winners into the benchmark configuration |
| `extract_imagenet100_from_tar.py` | the same pool from a local ImageNet tar instead of the mirror |
| `export_recon_bundle_only.py`, `add_corrupted_images_to_bundle.py` | package a run's reconstructions for sharing |
| `cluster_modules.sh` | cluster-conditional CUDA/cuDNN module selection, sourced by `setup_cluster.sh` |

Nothing here trains or reconstructs: `run.py` is the only entry point that needs a GPU.

## The search loop

There are no numbered stages. One round is run, `propose` reads it and writes the next, and
the loop ends when `propose` reports that no cell wants to move.

```bash
python run.py --config configs/experiments_hpo_grid.yaml --models pmf --run-id hpo_pmf --no-figures
python scripts/hpo.py propose --from outputs/hpo_pmf --models pmf --out configs/round2.yaml
python run.py --config configs/round2.yaml --run-id hpo_pmf_r2 --no-figures
python scripts/hpo.py propose --from outputs/hpo_pmf_r2 --grid outputs/hpo_pmf outputs/hpo_pmf_r2 \
    --top 0 --no-paper --models pmf --out configs/round3.yaml        # only cells still on an edge
python scripts/hpo.py finalize --from outputs/hpo_pmf_r2 --grid outputs/hpo_pmf \
    --models pmf --out configs/experiments_final_frozen100.yaml
```

`--from` is what the ranking reads; `--grid` is the union of everything tried so far and
decides what counts as a grid edge. Passing the original grid to `finalize` keeps the winner
inside the pre-registered ranges, which matters because on several axes (D-Flow and PnP step
counts, RHSO `N` and `M`) LPIPS keeps improving with compute far past the grid — a
compute-budget effect rather than a hyperparameter optimum.

Only the first round's configuration is committed. Later rounds are a function of results, so
they are generated rather than stored.

## 1000-image benchmark

One validation image per ImageNet-1k class, next to the paper's 100-image benchmark, which
stays byte-for-byte unchanged (files, configs and job ids).

```bash
# 1. the selection -- offline, already committed; rerun only to regenerate or to draw another
python scripts/make_frozen_selection.py                  # -> benchmarks/imagenet1000_c42_i43/
# 2. the image folder, from the ungated mirror (login node or laptop)
python scripts/build_local_imagenet_pool.py \
    --frozen-manifest benchmarks/imagenet1000_c42_i43/manifest.csv \
    --out cache/data/imagenet1000_c42_i43_mirror
# 3. the runs
python run.py --config configs/experiments_final_frozen1000.yaml --dry-run
python run.py --config configs/experiments_final_frozen1000.yaml --models pmf --run-id final1000_pmf --no-figures
```

**Selection.** The same draw as the 100-image benchmark — classes from
`numpy.random.default_rng(class_seed)`, one rank in 0–49 per class from
`numpy.random.default_rng(image_seed)`, class seed 42 / image seed 43 — plus one rule: a row the
**tuning pool** holds (`cache/data/imagenet_val_100`, seed 0: the first mirror row, rank 49, of
each of its 100 classes) is never selected, and that class is re-drawn from the same generator.
`selection_plan.json` lists the excluded rows and the re-draws. When the tuning pool is not on
disk its classes are reconstructed by the rule of `build_local_imagenet_pool.choose(100, seed=0)`,
and cross-checked against the manifest when it is. `manifest.csv` keeps the 100-image columns;
`original_archive_member` and `sha256` need the gated archive and are blank, so `filename` is
`<class>_<synset>_r<rank>.JPEG`, with synsets from `benchmarks/imagenet_class_index.json`.

**Configs.** `configs/experiments_final_frozen1000.yaml` carries the selected hyperparameters of
`experiments_final_frozen100.yaml` unchanged, with `num_images: 1000` and the new pool. It was
derived from the 100-image file because the HPO run directories are not in the checkout;
`hpo.py finalize --num-images 1000 --pool cache/data/imagenet1000_c42_i43_mirror` writes the
same file from them. `experiments_theory_N_fixed{B,M}_final1000.yaml` are the same copies of
the `N` sweeps.

**Runtime.** Roughly 10× per job. Resume is per finished job, so the SLURM walltime must cover
the longest single job (`bash submit.sh --config <file> --time ...`, or `--array N`).
`num_images` is in every job id: nothing collides with a 100-image run.
