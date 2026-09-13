# Final benchmark: 4 models x 5 problems x 6 methods on the paper's frozen ImageNet-100

Run 2026-09-10/11 on 4x A100-80GB, one model per GPU, one process per GPU (SiT was split into
4 shards, one per GPU, for its last 21 cells).  Configs: `configs/experiments_final_frozen100_<model>.yaml`.

* pool `cache/data/imagenet100_c42_i43_mirror` (upstream class seed 42 / image seed 43, rebuilt from the
  ungated mirror; degraded deblur / 2x-SR PSNR 25.91 / 22.81 vs the paper's 25.97 / 22.80)
* t0 = 1.0, beta = 1, seed 42, replicate 0, 100 images per cell, `save_individual_images: true`
* every cell uses the Stage-2 winner restricted to the pre-registered Stage-1 grid
  (`scripts/make_final_frozen100_configs.py --grid-only`); see docs/best_hyperparameters.md

Files
* `final_<model>/results.csv`, `results_per_image.csv` -- copies of `outputs/final_<model>/` after `--aggregate`
* `tables.md` -- LPIPS / PSNR / SSIM / missing-PSNR / s-per-image per cell (`scripts/make_final_tables.py`)
* `paired_lpips_vs_rhso.md`, `paired_psnr_vs_rhso.md`, `paired_lpips_vs_sdedit.md` -- paired per-image tests
  (`scripts/paired_tests.py`): same images, measurement, mask and epsilon inside a cell

Runtime column: JiT, pMF, iMF and the first 9 SiT cells ran alone on their GPU; the SiT cells after
2026-09-10 19:30 ran one process per GPU as well.  These are the only clean timings in this repository's
history; every Stage-1/2 runtime was measured with several processes sharing a GPU.

Paper comparison: the manuscript's Table 1 (pMF) and Table 8 (JiT) values are not stored in this repository;
`scripts/compare_to_paper.py` takes them as a CSV (`model,problem,method,lpips,psnr,ssim`) and prints the
cell-by-cell deltas against `final_<model>/results.csv`.
