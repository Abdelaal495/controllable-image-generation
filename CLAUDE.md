# CLAUDE.md

Conventions for working in this repository that are not derivable from the code alone.

## Frozen benchmarks

- Two frozen ImageNet-1k validation benchmarks live under `benchmarks/`, named
  `imagenet<N>_c<class seed>_i<image seed>`: `imagenet100_c42_i43` (the paper's 100 images)
  and `imagenet1000_c42_i43` (one image per class). Both use class seed 42 / image seed 43.
- **The 100-image benchmark is frozen.** Its files, `configs/*_final100.yaml`,
  `configs/experiments_final_frozen100.yaml` and their job ids must stay byte-for-byte
  unchanged. Add a benchmark by adding files, never by editing these.
- A selection is (class_id, within_class_validation_rank) pairs. The ungated mirror row is
  `class_id*50 + (49 - rank)`; the seeded pools (`--num-classes`) take the first mirror row
  per class, i.e. rank 49. `scripts/make_frozen_selection.py` draws a selection offline with
  numpy's `default_rng(class_seed).choice` and `default_rng(image_seed).integers(0, 50)`,
  the procedure that produced the 100-image plan (it reproduces that plan with
  `--num-classes 100 --no-exclude`).
- **Tuning-pool exclusion.** Hyperparameters were searched on the seed-0 pool
  `cache/data/imagenet_val_100`. A frozen benchmark drawn after that excludes every
  (class, mirror row) of that pool and records the excluded rows and re-draws in
  `selection_plan.json`. The pool is gitignored; when absent, its classes are reconstructed
  by the rule of `build_local_imagenet_pool.choose(100, seed=0)`.
- `manifest.csv` always has the columns `filename, class_id, synset, class_name,
  within_class_validation_rank, original_archive_member, sha256`. The last two need the gated
  archive; a benchmark drawn offline leaves them blank and names files
  `<class:03d>_<synset>_r<rank:02d>.JPEG`. Synsets and names come from
  `benchmarks/imagenet_class_index.json` (the standard 1000-class index).
- Image pools are built by `scripts/build_local_imagenet_pool.py --frozen-manifest
  <manifest.csv> --out cache/data/<manifest dir>_mirror`. **One benchmark per folder**:
  `src.data.DataManager._load_local` reads `sorted(files)[:n]` and refuses `n` above the
  count in the folder's `pool_manifest.json`.
- `num_images` is part of every job id (`src/config.py`), so 100- and 1000-image jobs never
  collide or resume from each other.

## Configurations

- `configs/experiments_final_frozen<N>.yaml` is written by `scripts/hpo.py finalize
  --num-images <N> --pool cache/data/imagenet<N>_c42_i43_mirror`. The 1000-image file was
  derived from the 100-image file (same winners, only `num_images` and `data.local_folder`
  differ) because the HPO run directories are not in the checkout; say so in the header when
  you do the same.
- 1000-image copies of theory configs are named `*_final1000.yaml`, change only
  `num_images`, `data.local_folder` and the header comments, and state the runtime caveat:
  roughly 10x per job, resume is per finished job, SLURM walltime must cover the longest job.

## Tests and docs

- `tests/` are plain scripts, not pytest: a `check()` helper, `PASSED`/`FAILED`, a `main()`
  that prints `N/N checks passed` and returns non-zero on failure. Benchmark tests must not
  download anything (set `HF_HUB_OFFLINE=1`, never import `huggingface_hub`/`datasets`).
- When a benchmark or config is added, update `README.md`, `scripts/README.md`,
  `tests/README.md`, and say which benchmark (100- or 1000-image) any doc sentence refers to
  (`docs/clusters_narval_rorqual.md`, `docs/schedule_and_rhso.md`).
- Solver code under `src/` is not touched for benchmark work beyond data loading.
- `run.py` calls `os.uname()` in `src/utils.py`, so `--dry-run` needs Linux/macOS (or a
  shim); Windows is not a supported runtime.
