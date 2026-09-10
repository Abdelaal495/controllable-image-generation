# Best hyperparameters per model, strategy and inverse problem at t0 = 1.0

Generated 2026-09-10 by `scripts/make_best_hyperparameters_t1.py` from Stage 2 (outputs/stage2_jit, outputs/stage2_sit, outputs/stage2_pmf, outputs/stage2_imf) and Stage 1 (outputs/hpo_jit, outputs/hpo_sit, outputs/hpo_pmf, outputs/hpo_imf).

Every entry is the lowest-mean-LPIPS configuration of its cell among the Stage-2 candidates (Stage-1 top three on 4 images, the manuscript's configuration where one exists, and the grid-edge extensions), re-run on **8 images** of `cache/data/imagenet_val_100`, `t0 = 1.0`, `beta = 1`, seed 42, replicate 0.  `+/- se` is the standard error of LPIPS over those 8 images.  `edge` marks a winner whose Stage-2 candidate set still had it on the extreme of an axis (see the Stage-2 generator's log).  For the two inpainting problems read `missing_psnr` in results.csv alongside these full-image metrics.

**Do not mix with `docs/best_hyperparameters.md`, which is the t0 = 0.8 study.**

**Selection rule.** The winner is the lowest-LPIPS configuration among the Stage-2 candidates that lie INSIDE the pre-registered Stage-1 grid (itself one step wider than the manuscript's ranges on every axis).  Configurations from the grid-edge extension rounds are excluded from selection; they are reported in the appendix at the end because on several axes (D-Flow steps, PnP steps, RHSO N/M) LPIPS kept improving with compute far past the grid, which is a compute-budget effect rather than a hyperparameter optimum.

---

# JiT-B/16 (pixel, standard flow, PyTorch)

## Denoising (sigma 0.20)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7152` | `0.0261` | `9.22` | `-0.0105` | 3 | `steps=25` |
| PnP-Flow | `0.0986` | `0.0120` | `28.00` | `0.7813` | 3 | `num_pnp_steps=50` `gamma0=200000.0` `alpha=0.5` |
| D-Flow | `0.4802` | `0.0437` | `23.38` | `0.5277` | 4 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.1144` | `0.0193` | `28.29` | `0.7875` | 3 | `K=2` `lam=15` `lr=0.05` |
| MPC-Delta_t | `0.1854` | `0.0240` | `27.31` | `0.7102` | 4 | `num_mpc_steps=8` `n_ctrl=20` `lam=30` `lr=0.3` |
| RHSO | `0.1343` | `0.0183` | `27.41` | `0.7838` | 4 | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.005` `mu=0.2` |

**best non-baseline:** PnP-Flow (LPIPS 0.0986)

## Deblurring (Gaussian 7/1.0, sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7152` | `0.0261` | `9.22` | `-0.0105` | 3 | `steps=25` |
| PnP-Flow | `0.1720` | `0.0434` | `28.28` | `0.8327` | 4 | `num_pnp_steps=200` `gamma0=1200000.0` `alpha=0.5` |
| D-Flow | `0.5283` | `0.0404` | `22.72` | `0.4899` | 4 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.1672` | `0.0270` | `28.26` | `0.7939` | 3 | `K=1` `lam=60` `lr=0.05` |
| MPC-Delta_t | `0.1445` | `0.0292` | `28.68` | `0.8049` | 4 | `num_mpc_steps=8` `n_ctrl=40` `lam=1000` `lr=0.1` |
| RHSO | `0.0965` | `0.0254` | `29.04` | `0.8488` | 4 | `num_rhso_steps=8` `num_opt_steps=20` `lr=0.01` `mu=0` |

**best non-baseline:** RHSO (LPIPS 0.0965)

## 2x super-resolution (sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7152` | `0.0261` | `9.22` | `-0.0105` | 3 | `steps=25` |
| PnP-Flow | `0.0937` | `0.0207` | `26.24` | `0.7870` | 4 | `num_pnp_steps=20` `gamma0=200000.0` `alpha=0.25` |
| D-Flow | `0.4804` | `0.0439` | `23.28` | `0.5300` | 4 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.1491` | `0.0286` | `26.11` | `0.7467` | 4 | `K=2` `lam=720` `lr=0.05` |
| MPC-Delta_t | `0.1007` | `0.0262` | `26.30` | `0.7758` | 4 | `num_mpc_steps=8` `n_ctrl=40` `lam=540` `lr=0.05` |
| RHSO | `0.1065` | `0.0291` | `26.70` | `0.7965` | 4 | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.01` `mu=0.2` |

**best non-baseline:** PnP-Flow (LPIPS 0.0937)

## Random inpainting (70% missing, sigma 0.01)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7152` | `0.0261` | `9.22` | `-0.0105` | 3 | `steps=25` |
| PnP-Flow | `0.0974` | `0.0178` | `26.42` | `0.7954` | 3 | `num_pnp_steps=50` `gamma0=100000.0` `alpha=0.1` |
| D-Flow | `0.4882` | `0.0422` | `23.05` | `0.5198` | 4 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.1717` | `0.0310` | `25.80` | `0.7343` | 4 | `K=2` `lam=240` `lr=0.02` |
| MPC-Delta_t | `0.0798` | `0.0182` | `27.06` | `0.8247` | 4 | `num_mpc_steps=8` `n_ctrl=20` `lam=1000` `lr=0.05` |
| RHSO | `0.0919` | `0.0282` | `26.99` | `0.8220` | 3 | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.01` `mu=0` |

**best non-baseline:** MPC-Delta_t (LPIPS 0.0798)

## Box inpainting (40x40, sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7152` | `0.0261` | `9.22` | `-0.0105` | 3 | `steps=25` |
| PnP-Flow | `0.0441` | `0.0052` | `30.61` | `0.8993` | 4 | `num_pnp_steps=50` `gamma0=200000.0` `alpha=0.1` |
| D-Flow | `0.4926` | `0.0436` | `21.25` | `0.5299` | 4 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.0635` | `0.0078` | `29.14` | `0.8581` | 3 | `K=1` `lam=30` `lr=0.05` |
| MPC-Delta_t | `0.0401` | `0.0043` | `30.92` | `0.8992` | 4 | `num_mpc_steps=8` `n_ctrl=20` `lam=180` `lr=0.1` |
| RHSO | `0.0458` | `0.0071` | `29.38` | `0.9079` | 3 | `num_rhso_steps=8` `num_opt_steps=20` `lr=0.01` `mu=0` |

**best non-baseline:** MPC-Delta_t (LPIPS 0.0401)

---

# SiT-XL/2 (latent, standard flow, PyTorch)

## Denoising (sigma 0.20)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7143` | `0.0213` | `9.72` | `0.0202` | 3 | `steps=1` |
| PnP-Flow | `0.4621` | `0.0471` | `24.42` | `0.6369` | 3 | `num_pnp_steps=200` `gamma0=100000.0` `alpha=0.5` |
| D-Flow | `0.3068` | `0.0328` | `24.43` | `0.6111` | 3 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.2563` | `0.0355` | `25.74` | `0.6802` | 3 | `K=1` `lam=5` `lr=0.2` |
| MPC-Delta_t | `0.2104` | `0.0317` | `26.05` | `0.6999` | 3 | `num_mpc_steps=8` `n_ctrl=40` `lam=180` `lr=0.3` |
| RHSO | `0.2287` | `0.0363` | `26.21` | `0.7034` | 3 | `num_rhso_steps=8` `num_opt_steps=80` `lr=0.03` `mu=0` |

**best non-baseline:** MPC-Delta_t (LPIPS 0.2104)

## Deblurring (Gaussian 7/1.0, sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7143` | `0.0213` | `9.72` | `0.0202` | 3 | `steps=1` |
| PnP-Flow | `0.4926` | `0.0470` | `24.07` | `0.6237` | 3 | `num_pnp_steps=200` `gamma0=100000.0` `alpha=0.5` |
| D-Flow | `0.3685` | `0.0236` | `23.16` | `0.5468` | 3 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.2890` | `0.0459` | `25.75` | `0.7035` | 3 | `K=1` `lam=720` `lr=0.2` |
| MPC-Delta_t | `0.2275` | `0.0297` | `25.72` | `0.6996` | 3 | `num_mpc_steps=8` `n_ctrl=40` `lam=360` `lr=0.3` |
| RHSO | `0.2305` | `0.0355` | `25.75` | `0.7183` | 3 | `num_rhso_steps=8` `num_opt_steps=80` `lr=0.03` `mu=0` |

**best non-baseline:** MPC-Delta_t (LPIPS 0.2275)

## 2x super-resolution (sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7143` | `0.0213` | `9.72` | `0.0202` | 3 | `steps=1` |
| PnP-Flow | `0.4608` | `0.0546` | `24.55` | `0.6297` | 3 | `num_pnp_steps=200` `gamma0=200000.0` `alpha=1` |
| D-Flow | `0.3199` | `0.0365` | `23.61` | `0.5837` | 3 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.2458` | `0.0400` | `25.50` | `0.6966` | 3 | `K=1` `lam=720` `lr=0.2` |
| MPC-Delta_t | `0.2203` | `0.0345` | `25.11` | `0.6857` | 3 | `num_mpc_steps=4` `n_ctrl=40` `lam=540` `lr=0.3` |
| RHSO | `0.2356` | `0.0404` | `24.58` | `0.6710` | 3 | `num_rhso_steps=8` `num_opt_steps=80` `lr=0.03` `mu=0` |

**best non-baseline:** MPC-Delta_t (LPIPS 0.2203)

## Random inpainting (70% missing, sigma 0.01)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7143` | `0.0213` | `9.72` | `0.0202` | 3 | `steps=1` |
| PnP-Flow | `0.4722` | `0.0513` | `24.23` | `0.6187` | 3 | `num_pnp_steps=200` `gamma0=200000.0` `alpha=1` |
| D-Flow | `0.3076` | `0.0312` | `23.78` | `0.5961` | 3 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.2457` | `0.0381` | `25.18` | `0.6838` | 3 | `K=1` `lam=60` `lr=0.2` |
| MPC-Delta_t | `0.2117` | `0.0320` | `24.81` | `0.6813` | 3 | `num_mpc_steps=8` `n_ctrl=40` `lam=1000` `lr=0.3` |
| RHSO | `0.2061` | `0.0364` | `24.99` | `0.6937` | 3 | `num_rhso_steps=8` `num_opt_steps=80` `lr=0.03` `mu=0` |

**best non-baseline:** RHSO (LPIPS 0.2061)

## Box inpainting (40x40, sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7143` | `0.0213` | `9.72` | `0.0202` | 3 | `steps=1` |
| PnP-Flow | `0.4675` | `0.0451` | `24.18` | `0.6362` | 3 | `num_pnp_steps=200` `gamma0=200000.0` `alpha=0.75` |
| D-Flow | `0.3022` | `0.0354` | `23.86` | `0.6260` | 3 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.2525` | `0.0424` | `25.86` | `0.7159` | 3 | `K=1` `lam=15` `lr=0.2` |
| MPC-Delta_t | `0.1942` | `0.0299` | `26.17` | `0.7440` | 3 | `num_mpc_steps=8` `n_ctrl=40` `lam=360` `lr=0.3` |
| RHSO | `0.2026` | `0.0350` | `25.83` | `0.7551` | 3 | `num_rhso_steps=8` `num_opt_steps=80` `lr=0.03` `mu=0` |

**best non-baseline:** MPC-Delta_t (LPIPS 0.1942)

---

# pMF-L/16 (pixel, MeanFlow, JAX)

## Denoising (sigma 0.20)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7136` | `0.0201` | `9.35` | `0.0189` | 3 | `steps=2` |
| PnP-Flow | `0.0787` | `0.0111` | `27.31` | `0.8100` | 4 | `num_pnp_steps=10` `gamma0=200000.0` `alpha=0.1` |
| D-Flow | `0.2101` | `0.0253` | `26.34` | `0.7068` | 4 | `num_opt_steps=320` `lr=0.1` |
| MPC-RHC | `0.1052` | `0.0202` | `28.18` | `0.7931` | 3 | `K=2` `lam=30` `lr=0.1` |
| MPC-Delta_t | `0.1670` | `0.0203` | `27.38` | `0.7216` | 4 | `num_mpc_steps=8` `n_ctrl=20` `lam=30` `lr=0.15` |
| RHSO | `0.0905` | `0.0150` | `28.32` | `0.7975` | 4 | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.03` `mu=1` |

**best non-baseline:** PnP-Flow (LPIPS 0.0787)

## Deblurring (Gaussian 7/1.0, sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7136` | `0.0201` | `9.35` | `0.0189` | 3 | `steps=2` |
| PnP-Flow | `0.1389` | `0.0396` | `27.33` | `0.8041` | 4 | `num_pnp_steps=200` `gamma0=5000000.0` `alpha=1` |
| D-Flow | `0.2901` | `0.0499` | `24.82` | `0.6700` | 3 | `num_opt_steps=640` `lr=0.03` |
| MPC-RHC | `0.1524` | `0.0315` | `28.09` | `0.7918` | 3 | `K=1` `lam=60` `lr=0.2` |
| MPC-Delta_t | `0.1352` | `0.0215` | `28.62` | `0.8034` | 4 | `num_mpc_steps=8` `n_ctrl=40` `lam=1000` `lr=0.1` |
| RHSO | `0.1211` | `0.0249` | `29.00` | `0.8473` | 3 | `num_rhso_steps=8` `num_opt_steps=20` `lr=0.01` `mu=0` |

**best non-baseline:** RHSO (LPIPS 0.1211)

## 2x super-resolution (sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7136` | `0.0201` | `9.35` | `0.0189` | 3 | `steps=2` |
| PnP-Flow | `0.1286` | `0.0239` | `25.24` | `0.7535` | 4 | `num_pnp_steps=20` `gamma0=100000.0` `alpha=0.1` |
| D-Flow | `0.1891` | `0.0311` | `25.56` | `0.7263` | 3 | `num_opt_steps=640` `lr=0.03` |
| MPC-RHC | `0.2027` | `0.0184` | `25.20` | `0.7099` | 3 | `K=3` `lam=180` `lr=0.05` |
| MPC-Delta_t | `0.0938` | `0.0214` | `26.52` | `0.7802` | 4 | `num_mpc_steps=8` `n_ctrl=40` `lam=360` `lr=0.15` |
| RHSO | `0.1072` | `0.0288` | `26.66` | `0.7931` | 4 | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.01` `mu=0.2` |

**best non-baseline:** MPC-Delta_t (LPIPS 0.0938)

## Random inpainting (70% missing, sigma 0.01)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7136` | `0.0201` | `9.35` | `0.0189` | 3 | `steps=2` |
| PnP-Flow | `0.1221` | `0.0183` | `25.23` | `0.7499` | 3 | `num_pnp_steps=20` `gamma0=100000.0` `alpha=0.1` |
| D-Flow | `0.2955` | `0.0929` | `24.44` | `0.6379` | 3 | `num_opt_steps=640` `lr=0.03` |
| MPC-RHC | `0.2217` | `0.0206` | `24.92` | `0.6892` | 3 | `K=3` `lam=180` `lr=0.05` |
| MPC-Delta_t | `0.0878` | `0.0204` | `26.95` | `0.8135` | 4 | `num_mpc_steps=8` `n_ctrl=40` `lam=360` `lr=0.15` |
| RHSO | `0.1026` | `0.0294` | `26.80` | `0.8145` | 4 | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.01` `mu=0` |

**best non-baseline:** MPC-Delta_t (LPIPS 0.0878)

## Box inpainting (40x40, sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7136` | `0.0201` | `9.35` | `0.0189` | 3 | `steps=2` |
| PnP-Flow | `0.0449` | `0.0051` | `29.63` | `0.8916` | 4 | `num_pnp_steps=50` `gamma0=200000.0` `alpha=0.1` |
| D-Flow | `0.1406` | `0.0177` | `24.67` | `0.8077` | 4 | `num_opt_steps=640` `lr=0.1` |
| MPC-RHC | `0.0810` | `0.0065` | `25.65` | `0.8542` | 3 | `K=1` `lam=30` `lr=0.05` |
| MPC-Delta_t | `0.0457` | `0.0059` | `29.04` | `0.8953` | 4 | `num_mpc_steps=8` `n_ctrl=40` `lam=180` `lr=0.3` |
| RHSO | `0.0485` | `0.0121` | `29.44` | `0.9010` | 4 | `num_rhso_steps=8` `num_opt_steps=20` `lr=0.01` `mu=0` |

**best non-baseline:** PnP-Flow (LPIPS 0.0449)

---

# iMF-B-2 (latent, MeanFlow, JAX)

## Denoising (sigma 0.20)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7558` | `0.0511` | `9.75` | `0.0498` | 3 | `steps=8` |
| PnP-Flow | `0.3597` | `0.0464` | `25.11` | `0.6398` | 3 | `num_pnp_steps=200` `gamma0=100000.0` `alpha=1` |
| D-Flow | `0.3657` | `0.0555` | `24.65` | `0.6258` | 3 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.2651` | `0.0508` | `26.03` | `0.6952` | 3 | `K=1` `lam=15` `lr=0.2` |
| MPC-Delta_t | `0.2357` | `0.0518` | `26.29` | `0.7125` | 3 | `num_mpc_steps=8` `n_ctrl=40` `lam=360` `lr=0.3` |
| RHSO | `0.2271` | `0.0468` | `26.36` | `0.7112` | 3 | `num_rhso_steps=8` `num_opt_steps=80` `lr=0.03` `mu=0` |

**best non-baseline:** RHSO (LPIPS 0.2271)

## Deblurring (Gaussian 7/1.0, sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7558` | `0.0511` | `9.75` | `0.0498` | 3 | `steps=8` |
| PnP-Flow | `0.4113` | `0.0399` | `24.52` | `0.6108` | 3 | `num_pnp_steps=200` `gamma0=100000.0` `alpha=1` |
| D-Flow | `0.3882` | `0.0482` | `24.24` | `0.6268` | 3 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.3011` | `0.0538` | `25.85` | `0.7089` | 3 | `K=1` `lam=180` `lr=0.2` |
| MPC-Delta_t | `0.2567` | `0.0606` | `26.14` | `0.7301` | 3 | `num_mpc_steps=8` `n_ctrl=40` `lam=1000` `lr=0.3` |
| RHSO | `0.2467` | `0.0562` | `25.90` | `0.7249` | 3 | `num_rhso_steps=8` `num_opt_steps=80` `lr=0.03` `mu=0` |

**best non-baseline:** RHSO (LPIPS 0.2467)

## 2x super-resolution (sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7558` | `0.0511` | `9.75` | `0.0498` | 3 | `steps=8` |
| PnP-Flow | `0.3570` | `0.0449` | `24.93` | `0.6382` | 3 | `num_pnp_steps=200` `gamma0=100000.0` `alpha=1` |
| D-Flow | `0.3594` | `0.0503` | `24.03` | `0.6114` | 3 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.2412` | `0.0482` | `25.60` | `0.7029` | 3 | `K=1` `lam=240` `lr=0.2` |
| MPC-Delta_t | `0.2376` | `0.0490` | `25.18` | `0.6983` | 3 | `num_mpc_steps=8` `n_ctrl=40` `lam=1000` `lr=0.3` |
| RHSO | `0.2360` | `0.0395` | `24.59` | `0.6748` | 3 | `num_rhso_steps=8` `num_opt_steps=80` `lr=0.03` `mu=0` |

**best non-baseline:** RHSO (LPIPS 0.2360)

## Random inpainting (70% missing, sigma 0.01)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7558` | `0.0511` | `9.75` | `0.0498` | 3 | `steps=8` |
| PnP-Flow | `0.3722` | `0.0449` | `24.57` | `0.6256` | 3 | `num_pnp_steps=200` `gamma0=100000.0` `alpha=1` |
| D-Flow | `0.3720` | `0.0487` | `23.72` | `0.6052` | 3 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.2587` | `0.0526` | `25.27` | `0.6953` | 3 | `K=1` `lam=180` `lr=0.2` |
| MPC-Delta_t | `0.2347` | `0.0551` | `25.10` | `0.6976` | 3 | `num_mpc_steps=8` `n_ctrl=40` `lam=1000` `lr=0.3` |
| RHSO | `0.2301` | `0.0513` | `25.03` | `0.7013` | 3 | `num_rhso_steps=8` `num_opt_steps=80` `lr=0.03` `mu=0` |

**best non-baseline:** RHSO (LPIPS 0.2301)

## Box inpainting (40x40, sigma 0.05)

| strategy | LPIPS | +/- se | PSNR | SSIM | n cand. | hyperparameters |
|---|---|---|---|---|---|---|
| SDEdit (baseline) | `0.7558` | `0.0511` | `9.75` | `0.0498` | 3 | `steps=8` |
| PnP-Flow | `0.3632` | `0.0448` | `24.87` | `0.6391` | 3 | `num_pnp_steps=200` `gamma0=100000.0` `alpha=1` |
| D-Flow | `0.3689` | `0.0516` | `24.21` | `0.6358` | 3 | `num_opt_steps=640` `lr=0.3` |
| MPC-RHC | `0.2498` | `0.0518` | `26.25` | `0.7285` | 3 | `K=1` `lam=60` `lr=0.2` |
| MPC-Delta_t | `0.2193` | `0.0490` | `26.46` | `0.7529` | 3 | `num_mpc_steps=8` `n_ctrl=40` `lam=540` `lr=0.3` |
| RHSO | `0.2121` | `0.0425` | `26.38` | `0.7603` | 3 | `num_rhso_steps=8` `num_opt_steps=80` `lr=0.03` `mu=0` |

**best non-baseline:** RHSO (LPIPS 0.2121)

---

# Are the manuscript's hyperparameters (Tables 4-7) optimal?

Rank of the manuscript's configuration inside each cell, and its LPIPS gap to the cell's winner.  Stage 1 ranks among all grid configurations on 4 images; Stage 2 ranks among the surviving candidates on 8 images (the manuscript's configuration is always one of them).  A gap of 0.0000 means the manuscript's choice is the winner.

| model | problem | method | manuscript config | Stage-1 rank / gap (4 img) | Stage-2 rank / gap (8 img) | manuscript LPIPS (8 img) | our winner | winner LPIPS |
|---|---|---|---|---|---|---|---|---|
| jit | denoising | pnp | num_pnp_steps=50 gamma0=200000.0 alpha=0.5 | #2/175 (+0.0046) | #1/3 (+0.0000) | `0.0986` | `num_pnp_steps=50` `gamma0=200000.0` `alpha=0.5` | `0.0986` |
| jit | denoising | dflow | num_opt_steps=320 lr=0.03 | #9/15 (+0.2093) | #4/4 (+0.2394) | `0.7196` | `num_opt_steps=640` `lr=0.3` | `0.4802` |
| jit | denoising | mpc_rhc | K=2 lam=15 lr=0.05 | #1/84 (+0.0000) | #1/3 (+0.0000) | `0.1144` | `K=2` `lam=15` `lr=0.05` | `0.1144` |
| jit | denoising | mpc_delta_t | num_mpc_steps=8 n_ctrl=40 lam=30 lr=0.1 | #4/96 (+0.0032) | #4/4 (+0.0068) | `0.1922` | `num_mpc_steps=8` `n_ctrl=20` `lam=30` `lr=0.3` | `0.1854` |
| jit | denoising | rhso | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0 | #10/26 (+0.1058) | #4/4 (+0.1555) | `0.2898` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.005` `mu=0.2` | `0.1343` |
| jit | deblur | pnp | num_pnp_steps=100 gamma0=1200000.0 alpha=0.5 | #6/175 (+0.0281) | #4/4 (+0.0254) | `0.1974` | `num_pnp_steps=200` `gamma0=1200000.0` `alpha=0.5` | `0.1720` |
| jit | deblur | dflow | num_opt_steps=320 lr=0.03 | #9/15 (+0.1707) | #4/4 (+0.1939) | `0.7221` | `num_opt_steps=640` `lr=0.3` | `0.5283` |
| jit | deblur | mpc_rhc | K=1 lam=60 lr=0.1 | #3/84 (+0.0028) | #3/3 (+0.0017) | `0.1689` | `K=1` `lam=60` `lr=0.05` | `0.1672` |
| jit | deblur | mpc_delta_t | num_mpc_steps=8 n_ctrl=40 lam=540 lr=0.1 | #6/96 (+0.0203) | #4/4 (+0.0129) | `0.1574` | `num_mpc_steps=8` `n_ctrl=40` `lam=1000` `lr=0.1` | `0.1445` |
| jit | deblur | rhso | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0 | #4/26 (+0.0600) | #3/4 (+0.0418) | `0.1383` | `num_rhso_steps=8` `num_opt_steps=20` `lr=0.01` `mu=0` | `0.0965` |
| jit | super_resolution | pnp | num_pnp_steps=20 gamma0=100000.0 alpha=0.25 | #13/175 (+0.0523) | #4/4 (+0.0503) | `0.1440` | `num_pnp_steps=20` `gamma0=200000.0` `alpha=0.25` | `0.0937` |
| jit | super_resolution | dflow | num_opt_steps=320 lr=0.03 | #9/15 (+0.2110) | #4/4 (+0.2390) | `0.7194` | `num_opt_steps=640` `lr=0.3` | `0.4804` |
| jit | super_resolution | mpc_rhc | K=3 lam=240 lr=0.05 | #8/84 (+0.0300) | #4/4 (+0.0140) | `0.1632` | `K=2` `lam=720` `lr=0.05` | `0.1491` |
| jit | super_resolution | mpc_delta_t | num_mpc_steps=4 n_ctrl=20 lam=360 lr=0.1 | #21/96 (+0.0095) | #4/4 (+0.0070) | `0.1077` | `num_mpc_steps=8` `n_ctrl=40` `lam=540` `lr=0.05` | `0.1007` |
| jit | super_resolution | rhso | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0 | #5/26 (+0.0304) | #4/4 (+0.0298) | `0.1363` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.01` `mu=0.2` | `0.1065` |
| jit | random_inpaint | pnp | num_pnp_steps=20 gamma0=200000.0 alpha=0.25 | #2/175 (+0.0023) | #2/3 (+0.0024) | `0.0998` | `num_pnp_steps=50` `gamma0=100000.0` `alpha=0.1` | `0.0974` |
| jit | random_inpaint | dflow | num_opt_steps=320 lr=0.03 | #9/15 (+0.2070) | #4/4 (+0.2318) | `0.7200` | `num_opt_steps=640` `lr=0.3` | `0.4882` |
| jit | random_inpaint | mpc_rhc | K=3 lam=180 lr=0.05 | #7/84 (+0.0166) | #3/4 (+0.0069) | `0.1786` | `K=2` `lam=240` `lr=0.02` | `0.1717` |
| jit | random_inpaint | mpc_delta_t | num_mpc_steps=4 n_ctrl=20 lam=540 lr=0.1 | #18/96 (+0.0143) | #4/4 (+0.0092) | `0.0890` | `num_mpc_steps=8` `n_ctrl=20` `lam=1000` `lr=0.05` | `0.0798` |
| jit | random_inpaint | rhso | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0 | #2/26 (+0.0036) | #1/3 (+0.0000) | `0.0919` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.01` `mu=0` | `0.0919` |
| jit | box_inpaint | pnp | num_pnp_steps=100 gamma0=400000.0 alpha=0.25 | #10/175 (+0.0108) | #4/4 (+0.0119) | `0.0560` | `num_pnp_steps=50` `gamma0=200000.0` `alpha=0.1` | `0.0441` |
| jit | box_inpaint | dflow | num_opt_steps=320 lr=0.03 | #9/15 (+0.2058) | #4/4 (+0.2273) | `0.7199` | `num_opt_steps=640` `lr=0.3` | `0.4926` |
| jit | box_inpaint | mpc_rhc | K=1 lam=30 lr=0.05 | #1/84 (+0.0000) | #1/3 (+0.0000) | `0.0635` | `K=1` `lam=30` `lr=0.05` | `0.0635` |
| jit | box_inpaint | mpc_delta_t | num_mpc_steps=8 n_ctrl=40 lam=180 lr=0.1 | #4/96 (+0.0036) | #4/4 (+0.0023) | `0.0424` | `num_mpc_steps=8` `n_ctrl=20` `lam=180` `lr=0.1` | `0.0401` |
| jit | box_inpaint | rhso | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0 | #2/26 (+0.0122) | #2/3 (+0.0035) | `0.0492` | `num_rhso_steps=8` `num_opt_steps=20` `lr=0.01` `mu=0` | `0.0458` |
| pmf | denoising | pnp | num_pnp_steps=50 gamma0=800000.0 alpha=0.75 | #23/175 (+0.0466) | #4/4 (+0.0357) | `0.1144` | `num_pnp_steps=10` `gamma0=200000.0` `alpha=0.1` | `0.0787` |
| pmf | denoising | dflow | num_opt_steps=320 lr=0.03 | #5/15 (+0.1292) | #4/4 (+0.0878) | `0.2978` | `num_opt_steps=320` `lr=0.1` | `0.2101` |
| pmf | denoising | mpc_rhc | K=2 lam=15 lr=0.1 | #3/84 (+0.0179) | #3/3 (+0.0098) | `0.1150` | `K=2` `lam=30` `lr=0.1` | `0.1052` |
| pmf | denoising | mpc_delta_t | num_mpc_steps=8 n_ctrl=40 lam=30 lr=0.1 | #5/96 (+0.0025) | #3/4 (+0.0023) | `0.1693` | `num_mpc_steps=8` `n_ctrl=20` `lam=30` `lr=0.15` | `0.1670` |
| pmf | denoising | rhso | num_rhso_steps=4 num_opt_steps=40 lr=0.03 mu=0 | #13/26 (+0.0843) | #4/4 (+0.1037) | `0.1942` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.03` `mu=1` | `0.0905` |
| pmf | deblur | pnp | num_pnp_steps=100 gamma0=1200000.0 alpha=0.5 | #4/175 (+0.0487) | #3/4 (+0.0553) | `0.1942` | `num_pnp_steps=200` `gamma0=5000000.0` `alpha=1` | `0.1389` |
| pmf | deblur | dflow | num_opt_steps=320 lr=0.03 | #2/15 (+0.0517) | #2/3 (+0.0478) | `0.3380` | `num_opt_steps=640` `lr=0.03` | `0.2901` |
| pmf | deblur | mpc_rhc | K=1 lam=60 lr=0.05 | #3/84 (+0.0042) | #3/3 (+0.0025) | `0.1549` | `K=1` `lam=60` `lr=0.2` | `0.1524` |
| pmf | deblur | mpc_delta_t | num_mpc_steps=8 n_ctrl=40 lam=540 lr=0.1 | #7/96 (+0.0227) | #4/4 (+0.0104) | `0.1456` | `num_mpc_steps=8` `n_ctrl=40` `lam=1000` `lr=0.1` | `0.1352` |
| pmf | deblur | rhso | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0 | #2/26 (+0.0353) | #3/3 (+0.0323) | `0.1534` | `num_rhso_steps=8` `num_opt_steps=20` `lr=0.01` `mu=0` | `0.1211` |
| pmf | super_resolution | pnp | num_pnp_steps=20 gamma0=100000.0 alpha=0.25 | #6/175 (+0.0270) | #4/4 (+0.0177) | `0.1463` | `num_pnp_steps=20` `gamma0=100000.0` `alpha=0.1` | `0.1286` |
| pmf | super_resolution | dflow | num_opt_steps=320 lr=0.03 | #2/15 (+0.0992) | #2/3 (+0.0913) | `0.2804` | `num_opt_steps=640` `lr=0.03` | `0.1891` |
| pmf | super_resolution | mpc_rhc | K=3 lam=240 lr=0.05 | #2/84 (+0.0046) | #2/3 (+0.0016) | `0.2042` | `K=3` `lam=180` `lr=0.05` | `0.2027` |
| pmf | super_resolution | mpc_delta_t | num_mpc_steps=4 n_ctrl=20 lam=360 lr=0.1 | #39/96 (+0.0513) | #4/4 (+0.0320) | `0.1258` | `num_mpc_steps=8` `n_ctrl=40` `lam=360` `lr=0.15` | `0.0938` |
| pmf | super_resolution | rhso | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0 | #6/26 (+0.0245) | #4/4 (+0.0234) | `0.1306` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.01` `mu=0.2` | `0.1072` |
| pmf | random_inpaint | pnp | num_pnp_steps=20 gamma0=100000.0 alpha=0.25 | #2/175 (+0.0180) | #2/3 (+0.0196) | `0.1417` | `num_pnp_steps=20` `gamma0=100000.0` `alpha=0.1` | `0.1221` |
| pmf | random_inpaint | dflow | num_opt_steps=320 lr=0.03 | #2/15 (+0.1101) | #2/3 (+0.0243) | `0.3198` | `num_opt_steps=640` `lr=0.03` | `0.2955` |
| pmf | random_inpaint | mpc_rhc | K=3 lam=180 lr=0.05 | #1/84 (+0.0000) | #1/3 (+0.0000) | `0.2217` | `K=3` `lam=180` `lr=0.05` | `0.2217` |
| pmf | random_inpaint | mpc_delta_t | num_mpc_steps=4 n_ctrl=20 lam=540 lr=0.15 | #37/96 (+0.0349) | #4/4 (+0.0199) | `0.1077` | `num_mpc_steps=8` `n_ctrl=40` `lam=360` `lr=0.15` | `0.0878` |
| pmf | random_inpaint | rhso | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0 | #4/26 (+0.0094) | #1/4 (+0.0000) | `0.1026` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.01` `mu=0` | `0.1026` |
| pmf | box_inpaint | pnp | num_pnp_steps=100 gamma0=400000.0 alpha=0.25 | #10/175 (+0.0198) | #4/4 (+0.0159) | `0.0608` | `num_pnp_steps=50` `gamma0=200000.0` `alpha=0.1` | `0.0449` |
| pmf | box_inpaint | dflow | num_opt_steps=320 lr=0.03 | #6/15 (+0.1515) | #4/4 (+0.1680) | `0.3086` | `num_opt_steps=640` `lr=0.1` | `0.1406` |
| pmf | box_inpaint | mpc_rhc | K=1 lam=30 lr=0.05 | #1/84 (+0.0000) | #1/3 (+0.0000) | `0.0810` | `K=1` `lam=30` `lr=0.05` | `0.0810` |
| pmf | box_inpaint | mpc_delta_t | num_mpc_steps=8 n_ctrl=40 lam=180 lr=0.1 | #8/96 (+0.0081) | #2/4 (+0.0027) | `0.0483` | `num_mpc_steps=8` `n_ctrl=40` `lam=180` `lr=0.3` | `0.0457` |
| pmf | box_inpaint | rhso | num_rhso_steps=4 num_opt_steps=40 lr=0.02 mu=0 | #5/26 (+0.0359) | #4/4 (+0.0184) | `0.0670` | `num_rhso_steps=8` `num_opt_steps=20` `lr=0.01` `mu=0` | `0.0485` |

---

# Appendix: grid-edge extension rounds (NOT used for the final benchmark)

Lowest LPIPS reached in each cell when the edge of the grid was pushed repeatedly (8 images), next to the grid-only winner above.

| model | problem | method | grid-only winner LPIPS | extended best LPIPS | extended configuration |
|---|---|---|---|---|---|
| jit | denoising | dflow | `0.4802` | `0.2884` | `num_opt_steps=2560` `lr=0.3` |
| jit | deblur | dflow | `0.5283` | `0.3008` | `num_opt_steps=5120` `lr=0.3` |
| jit | deblur | mpc_delta_t | `0.1445` | `0.1396` | `num_mpc_steps=8` `n_ctrl=320` `lam=1000` `lr=0.1` |
| jit | super_resolution | dflow | `0.4804` | `0.3089` | `num_opt_steps=2560` `lr=0.3` |
| jit | random_inpaint | dflow | `0.4882` | `0.3137` | `num_opt_steps=2560` `lr=0.3` |
| jit | box_inpaint | dflow | `0.4926` | `0.1467` | `num_opt_steps=20480` `lr=0.3` |
| pmf | denoising | dflow | `0.2101` | `0.2054` | `num_opt_steps=2560` `lr=0.1` |
| pmf | deblur | dflow | `0.2901` | `0.1315` | `num_opt_steps=5120` `lr=0.03` |
| pmf | super_resolution | dflow | `0.1891` | `0.1383` | `num_opt_steps=2560` `lr=0.03` |
| pmf | random_inpaint | dflow | `0.2955` | `0.1484` | `num_opt_steps=2560` `lr=0.03` |
| imf | denoising | pnp | `0.3597` | `0.2249` | `num_pnp_steps=4004` `gamma0=100000.0` `alpha=1` |
| imf | denoising | dflow | `0.3657` | `0.2490` | `num_opt_steps=5120` `lr=0.3` |
| imf | denoising | mpc_delta_t | `0.2357` | `0.2200` | `num_mpc_steps=8` `n_ctrl=160` `lam=540` `lr=0.3` |
| imf | denoising | rhso | `0.2271` | `0.2197` | `num_rhso_steps=8` `num_opt_steps=320` `lr=0.03` `mu=0` |
| imf | deblur | pnp | `0.4113` | `0.2866` | `num_pnp_steps=4004` `gamma0=100000.0` `alpha=1` |
| imf | deblur | dflow | `0.3882` | `0.2776` | `num_opt_steps=5120` `lr=0.3` |
| imf | deblur | mpc_delta_t | `0.2567` | `0.2335` | `num_mpc_steps=8` `n_ctrl=320` `lam=1000` `lr=0.3` |
| imf | deblur | rhso | `0.2467` | `0.2418` | `num_rhso_steps=32` `num_opt_steps=80` `lr=0.03` `mu=0` |
| imf | super_resolution | pnp | `0.3570` | `0.1999` | `num_pnp_steps=8468` `gamma0=100000.0` `alpha=1` |
| imf | super_resolution | dflow | `0.3594` | `0.2535` | `num_opt_steps=5120` `lr=0.3` |
| imf | super_resolution | mpc_delta_t | `0.2376` | `0.2311` | `num_mpc_steps=8` `n_ctrl=40` `lam=6300` `lr=0.3` |
| imf | random_inpaint | pnp | `0.3722` | `0.2341` | `num_pnp_steps=4004` `gamma0=100000.0` `alpha=1` |
| imf | random_inpaint | dflow | `0.3720` | `0.2493` | `num_opt_steps=5120` `lr=0.3` |
| imf | random_inpaint | mpc_delta_t | `0.2347` | `0.2125` | `num_mpc_steps=8` `n_ctrl=160` `lam=1000` `lr=0.3` |
| imf | random_inpaint | rhso | `0.2301` | `0.1969` | `num_rhso_steps=8` `num_opt_steps=640` `lr=0.03` `mu=0` |
| imf | box_inpaint | pnp | `0.3632` | `0.2527` | `num_pnp_steps=4004` `gamma0=100000.0` `alpha=1` |
| imf | box_inpaint | dflow | `0.3689` | `0.2142` | `num_opt_steps=5120` `lr=0.3` |
| imf | box_inpaint | mpc_delta_t | `0.2193` | `0.1932` | `num_mpc_steps=8` `n_ctrl=320` `lam=1000` `lr=0.3` |
| imf | box_inpaint | rhso | `0.2121` | `0.1947` | `num_rhso_steps=32` `num_opt_steps=80` `lr=0.03` `mu=0` |
