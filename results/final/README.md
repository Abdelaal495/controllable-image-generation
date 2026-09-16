# Final benchmark: 4 models x 5 problems x 6 methods on the frozen ImageNet-100

Config `configs/experiments_final_frozen100.yaml`, run once per model with `--models <model>`,
one process per A100-80GB.

* pool `cache/data/imagenet100_c42_i43_mirror` (class seed 42 / image seed 43), rebuilt from the
  ungated mirror by `scripts/build_local_imagenet_pool.py --frozen-manifest benchmarks/imagenet100_c42_i43/manifest.csv`
* t0 = 1.0, beta = 1, seed 42, replicate 0, 100 images per cell
* every cell uses its Stage-2 winner restricted to the Stage-1 grid
  (`scripts/make_final_frozen100_configs.py --grid-only`); see `docs/best_hyperparameters.md`

Files

* `final_<model>/results.csv`, `results_per_image.csv` -- `outputs/final_<model>/` after `--aggregate`
* `tables.md` -- LPIPS / PSNR / SSIM / missing-region PSNR / s-per-image per cell (`scripts/make_final_tables.py`)
* `results.tex` -- the same numbers as LaTeX tables (`scripts/make_results_tex.py`)
* `paired_lpips_vs_rhso.md`, `paired_psnr_vs_rhso.md`, `paired_lpips_vs_sdedit.md` -- paired per-image
  tests, same images / measurement / mask / epsilon inside a cell (`scripts/paired_tests.py`)

Runtimes here were measured with one process per GPU.  The Stage-1 and Stage-2 runtimes under
`results/stage1` and `results/stage2` were measured with several processes sharing a GPU and are
only comparable within a run.
