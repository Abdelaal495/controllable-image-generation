# Provenance of the paper figures

Where every pixel in `figures/fig_qualitative_comparison.pdf` (Fig. 1) and
`figures/fig_rhso_trajectory_{box,random}_inpaint.pdf` (Fig. 2) comes from, so the figures can
be regenerated or defended without re-deriving any of it.

## Source images

Both figures draw from **our own 100-class pool**, `cache/data/imagenet_val_100`, built by

    python build_local_imagenet_pool.py --num-classes 100 --seed 0

from the ungated mirror `evanarlian/imagenet_1k_resized_256` (ImageNet-1k validation, short side
resized to 256).  The first 32 classes follow `src.data.IMAGENET_EXAMPLES` in order; the other 68
are drawn with seed 0.  For every class the pool takes the mirror's FIRST row of that class.  The
mirror's per-class blocks are in *reversed* validation-filename order (established 2026-09-06, see
`cache/data/imagenet100_c42_i43_mirror/pool_manifest.json`), so "first row" is the validation
image with within-class filename rank 49.  Loading applies `src.data.center_crop` (short side to
256, bicubic, central square).

Image ids are `<class>_<pool index>_<name>`; named ids are curated classes, `classNNN` ids are the
random draws.

| figure | row / use | image id | ImageNet class |
|---|---|---|---|
| Fig. 1 | denoising | `084_29_peacock` | 84, peacock |
| Fig. 1 | deblurring | `181_48_class181` | 181, Bedlington terrier |
| Fig. 1 | 2x super-resolution | `387_68_class387` | 387, lesser panda |
| Fig. 1 | random inpainting | `153_95_class153` | 153, Maltese dog |
| Fig. 1 | box inpainting | `143_76_class143` | 143, oystercatcher |
| Fig. 2 | RHSO trajectory (both variants) | `291_06_lion` | 291, lion (pool index 6) |

**None of the six is in the upstream frozen benchmark** `benchmarks/imagenet100_c42_i43`
(class seed 42, image seed 43): none of the six classes is among its 100 classes.  The frozen set,
rebuilt from the mirror as `cache/data/imagenet100_c42_i43_mirror`, is the set the quantitative
tables should be run on; the qualitative figures were chosen for legibility, not membership, and
the manuscript's own instruction was that the source image does not matter for them.

## How each panel was produced

* **Ground truth** -- the pool image after `center_crop`.
* **Measurement** -- the repository's degradation operators with the manuscript's constants
  (denoising sigma 0.20; deblur Gaussian sigma 1.0, kernel 7, reflect padding, sigma 0.05;
  super-resolution true 2x subsampling, sigma 0.05; box 40x40, sigma 0.05; random 70% missing,
  sigma 0.01), noise and masks seeded from `(global seed 42, problem parameters, image id)`.
* **Fig. 1 reconstructions** -- read from disk, from run `outputs/t1_final`
  (`configs/experiments_t1_final.yaml`: pMF, t0 = 1.0, 100 images, each method at the
  configuration that won `outputs/t1_hpo`).  No job was run for the figure.  PSNR under each panel
  is the value recorded in `results_per_image.csv` for that image.
* **Selection rule (Fig. 1)** -- among images where RHSO beats all four rivals on BOTH LPIPS and
  PSNR (`make_paper_figures.py --auto-select` prints the ranking), chosen by eye for a
  recognisable subject with detail in the range the degradation damages; no image reused
  between rows.  Pure argmax on the margin was rejected because it picks images whose statistics
  are unusual rather than legible (a mostly-white geyser, a fur close-up, a product photo covered
  in text).
* **Fig. 2 states and readouts** -- one dedicated traced run, `configs/figure_trajectory.yaml`
  (pMF, box and random inpainting, N = 4, M = 40, lr = 0.01, mu = 0 / 0.2 -- the winning
  settings at the time), with `src.rhso.TRACE` installed.  The trace records, per stage,
  x_{t_k}, the terminal readout before optimising and after optimising, and is cached as
  `cache/rhso_traces.npz` (uint8, regenerable with `--collect`).  Row 1 of the figure is the
  committed trajectory x_{t_0..4}; rows 2-3 are the readouts; the last readout of the last
  stage is the reconstruction (asserted in code).

## Regeneration

    python make_paper_figures.py --figure comparison --run outputs/t1_final \
        --config configs/experiments_t1_final.yaml
    python make_paper_figures.py --figure trajectory --problem box_inpaint      # or random_inpaint
    python make_paper_figures.py --figure trajectory --problem box_inpaint --pptx

To draw from the frozen upstream set instead, pass `--pool cache/data/imagenet100_c42_i43_mirror`
and a run made on it, and re-collect the trace with `--collect` on that pool.
