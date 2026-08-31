# Best hyperparameters per model, strategy and inverse problem

Generated 2026-08-31 from every tuning run in `outputs/` (1536 runs across six sweeps).

**120 tuned entries** = 4 models x 5 strategies x 6 inverse problems. SDEdit appears in
each table as the paired baseline but is not one of the 120: it has no measurement term
to tune, only a step count.

## How to read this

Each entry is the lowest-LPIPS configuration swept for that cell, at `t0 = 0.8` and
`beta = 1.0` unless stated, on 8 ImageNet validation images, seed 42, replicate 0.

- **`+/- se`** is the standard error of LPIPS across those 8 images.
- **`ties`** counts the other configurations in the same cell that a paired per-image
  t-test cannot separate from the winner (|t| < 2.36, two-sided 5% at 7 d.o.f.). A cell
  with several ties has a plateau rather than a point optimum: the exact winning value is
  partly luck and any tied configuration is an equally defensible choice.
- **`edge`** flags a winner sitting at the end of its swept range. Those are directions,
  not optima -- the real best value is somewhere outside what was tried.

Because every run in a cell shares bit-identical images and generative noise, the paired
comparison is much more sensitive than comparing means. It is still 8 images: enough to
separate a real effect from a coin flip, not enough to rank configurations differing in
the third decimal.

For the inpainting problems read `missing_psnr` in `results.csv` alongside LPIPS:
full-image metrics there are dominated by pixels the measurement already supplies.

## Two findings that override the shipped defaults

**PnP-Flow needs `phi_normalization: half_sum_squared`.** Its per-method default divides
the fidelity term by the measurement count (196,608 for a 256x256 RGB image), which
leaves `gamma0` inert -- LPIPS spread 0.018 across a 4x range of `gamma0` under the
default, versus 0.204 once the scale is fixed. On JiT this moved PnP from worst method
to best on denoising.

**RHSO's `mu` is state-space dependent, and the earlier "mu = 0" advice was wrong for
pixel models.** On SiT and iMF, `mu = 0` wins. On JiT and pMF, `mu > 0` wins in all 12
cells, by as much as 0.35 LPIPS (JiT deblurring: 0.497 at `mu=0` versus 0.151 at
`mu=0.2`). Ten of those twelve chose the largest value swept, so `mu` is still an edge
and the pixel entries below are a lower bound on what it is worth.

This mirrors what `lambda` does in MPC: a penalty on the optimised variable helps in
pixel space and does not in a latent space. See `configs/experiments_lambda_pixel_control.yaml`.

| run directory | configuration | scope |
|---|---|---|
| `outputs/hpo_v2`, `outputs/hpo_v3` | `experiments_sit_imf_hpo_v2.yaml`, `_v3.yaml` | SiT/iMF, denoising, 360 runs |
| `outputs/tasks5` | `experiments_sit_imf_tasks.yaml` | SiT/iMF, other five problems, 170 runs |
| `outputs/tasks_pixel` | `experiments_jit_pmf_tasks.yaml` | JiT/pMF, all six problems, 504 runs |
| `outputs/lambda_pixel` | `experiments_lambda_pixel_control.yaml` | four-model lambda control arm, 64 runs |
| `outputs/round4` | `experiments_round4.yaml` | lambda x n_ctrl, K=2, mu on pixel models, 342 runs |
| `outputs/round5` | `experiments_round5.yaml` | mu extended to 2.0 on all four models, 96 runs |

---

# JiT-B/16

pixel 3x256x256, standard flow, Torch

## JiT-B/16 &mdash; Denoising

Degraded observation: LPIPS `0.4712`, PSNR `20.46 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.1136` | `0.0180` | `27.74` | `0.7745` | 1 / 19 | K at max, lr at min | `num_mpc_steps=4` `K=3` `lam=5.0` `n_ctrl=20` `lr=0.05` |
| MPC-Delta_t | `0.1749` | `0.0187` | `26.61` | `0.6499` | 4 / 15 | n_ctrl at max | `num_mpc_steps=4` `lam=30.0` `n_ctrl=80` `lr=0.1` |
| PnP-Flow | `0.1050` | `0.0164` | `28.27` | `0.7983` | 0 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2272` | `0.0661` | `27.41` | `0.7516` | 1 / 3 | num_opt_steps at max, lr at min | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.1028` | `0.0185` | `28.49` | `0.7959` | 0 / 17 | num_opt_steps at max, lr at min, mu at max, beta at max | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=2.0` `beta=0.5` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.4152` | `0.0369` | `18.04` | `0.3266` | 1 / 1 | steps at max | `steps=25` `solver=heun` |

## JiT-B/16 &mdash; Deblurring

Degraded observation: LPIPS `0.2709`, PSNR `25.74 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.1568` | `0.0288` | `28.32` | `0.8069` | 7 / 19 | K at min, lr at max | `num_mpc_steps=4` `K=1` `lam=50.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.1643` | `0.0353` | `28.79` | `0.8185` | 2 / 15 | n_ctrl at max | `num_mpc_steps=4` `lam=300.0` `n_ctrl=80` `lr=0.1` |
| PnP-Flow | `0.3700` | `0.0530` | `25.65` | `0.6928` | 0 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2839` | `0.0539` | `26.22` | `0.7069` | 1 / 3 | num_opt_steps at max, lr at max | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.1` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.1468` | `0.0411` | `28.90` | `0.8449` | 4 / 17 | num_opt_steps at max, lr at min, beta at max | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.2` `beta=0.5` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.4268` | `0.0370` | `18.18` | `0.3263` | 1 / 1 | steps at max | `steps=25` `solver=heun` |

## JiT-B/16 &mdash; Super-resolution

Degraded observation: LPIPS `0.2324`, PSNR `22.73 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.1100` | `0.0296` | `26.88` | `0.7938` | 2 / 19 | K at max, lr at min | `num_mpc_steps=4` `K=3` `lam=150.0` `n_ctrl=20` `lr=0.05` |
| MPC-Delta_t | `0.1085` | `0.0287` | `26.12` | `0.7657` | 0 / 15 | — | `num_mpc_steps=4` `lam=300.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.4312` | `0.0493` | `23.16` | `0.5485` | 0 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2517` | `0.0754` | `26.24` | `0.7318` | 1 / 3 | num_opt_steps at max, lr at min | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.1036` | `0.0291` | `26.32` | `0.7864` | 0 / 17 | num_opt_steps at max, lr at min, beta at max | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=1.0` `beta=0.5` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.4153` | `0.0369` | `18.03` | `0.3253` | 1 / 1 | steps at max | `steps=25` `solver=heun` |

## JiT-B/16 &mdash; Box inpainting

Degraded observation: LPIPS `0.1010`, PSNR `27.50 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.0620` | `0.0123` | `28.08` | `0.8694` | 6 / 19 | K at max, lr at min | `num_mpc_steps=4` `K=3` `lam=50.0` `n_ctrl=20` `lr=0.05` |
| MPC-Delta_t | `0.0502` | `0.0061` | `29.05` | `0.8838` | 1 / 15 | — | `num_mpc_steps=4` `lam=100.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.1673` | `0.0306` | `27.36` | `0.8021` | 0 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2502` | `0.0642` | `26.08` | `0.7496` | 1 / 3 | num_opt_steps at max, lr at min | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.0364` | `0.0064` | `29.23` | `0.9082` | 1 / 17 | num_opt_steps at max, lr at min, beta at max | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.2` `beta=0.5` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.4181` | `0.0341` | `18.49` | `0.3505` | 1 / 1 | steps at min | `steps=4` `solver=heun` |

## JiT-B/16 &mdash; Random inpainting

Degraded observation: LPIPS `1.0188`, PSNR `12.72 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.1325` | `0.0232` | `26.24` | `0.7647` | 1 / 19 | K at max, lr at min | `num_mpc_steps=4` `K=3` `lam=150.0` `n_ctrl=20` `lr=0.05` |
| MPC-Delta_t | `0.0903` | `0.0223` | `26.37` | `0.8074` | 2 / 15 | lam at max | `num_mpc_steps=4` `lam=1000.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.3825` | `0.0512` | `23.61` | `0.5780` | 0 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2910` | `0.0498` | `25.46` | `0.6844` | 1 / 3 | num_opt_steps at max, lr at min | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.1003` | `0.0234` | `26.30` | `0.7998` | 5 / 17 | num_opt_steps at max, lr at min, beta at max | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.5` `beta=0.5` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.7014` | `0.0296` | `13.25` | `0.0997` | 1 / 1 | steps at max | `steps=25` `solver=heun` |

## JiT-B/16 &mdash; Stroke painting

Degraded observation: LPIPS `0.4517`, PSNR `19.63 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.4196` | `0.0385` | `19.24` | `0.3784` | 12 / 19 | K at max | `num_mpc_steps=4` `K=3` `lam=5.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.4296` | `0.0377` | `19.10` | `0.3769` | 13 / 15 | n_ctrl at min | `num_mpc_steps=4` `lam=15.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.5789` | `0.0466` | `19.31` | `0.4063` | 1 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.5247` | `0.0368` | `14.90` | `0.2702` | 1 / 3 | num_opt_steps at min, lr at min | `steps=2` `solver=heun` `num_opt_steps=40` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.4292` | `0.0403` | `17.60` | `0.3155` | 2 / 17 | num_opt_steps at max, lr at min, beta at max | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=1.0` `beta=0.5` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.4695` | `0.0362` | `17.44` | `0.2848` | 1 / 1 | steps at max | `steps=25` `solver=heun` |

---

# pMF-L-16

pixel 256x256x3, MeanFlow, JAX

## pMF-L-16 &mdash; Denoising

Degraded observation: LPIPS `0.4712`, PSNR `20.46 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.1068` | `0.0174` | `27.73` | `0.7712` | 0 / 19 | lr at max | `num_mpc_steps=2` `K=2` `lam=15.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.2501` | `0.0233` | `25.05` | `0.5925` | 2 / 15 | lam at min | `num_mpc_steps=2` `lam=5.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.1050` | `0.0172` | `27.55` | `0.7777` | 0 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2695` | `0.0676` | `25.44` | `0.6752` | 1 / 3 | num_opt_steps at max, lr at min | `steps=2` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.0988` | `0.0146` | `27.96` | `0.7792` | 0 / 17 | num_opt_steps at max, lr at min, mu at max, beta at max | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=2.0` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.3890` | `0.0480` | `18.49` | `0.3454` | 1 / 1 | steps at max | `steps=2` |

## pMF-L-16 &mdash; Deblurring

Degraded observation: LPIPS `0.2709`, PSNR `25.74 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.1791` | `0.0276` | `27.28` | `0.7508` | 4 / 19 | K at min, lr at max | `num_mpc_steps=2` `K=1` `lam=50.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.1818` | `0.0316` | `28.32` | `0.7960` | 10 / 15 | n_ctrl at max | `num_mpc_steps=2` `lam=100.0` `n_ctrl=80` `lr=0.1` |
| PnP-Flow | `0.2613` | `0.0527` | `24.33` | `0.6796` | 0 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2518` | `0.0526` | `25.71` | `0.7034` | 0 / 3 | num_opt_steps at max, lr at min | `steps=2` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.2305` | `0.0297` | `26.96` | `0.7626` | 10 / 17 | num_opt_steps at max, lr at min, beta at max | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=0.05` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.3944` | `0.0492` | `18.47` | `0.3390` | 1 / 1 | steps at max | `steps=2` |

## pMF-L-16 &mdash; Super-resolution

Degraded observation: LPIPS `0.2324`, PSNR `22.73 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.2337` | `0.0345` | `24.28` | `0.6013` | 7 / 19 | lr at max | `num_mpc_steps=2` `K=2` `lam=50.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.1153` | `0.0305` | `26.32` | `0.7701` | 4 / 15 | lam at max, n_ctrl at max | `num_mpc_steps=2` `lam=1000.0` `n_ctrl=80` `lr=0.1` |
| PnP-Flow | `0.2987` | `0.0300` | `22.35` | `0.5749` | 0 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2446` | `0.0593` | `25.40` | `0.6913` | 1 / 3 | num_opt_steps at max, lr at min | `steps=2` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.1167` | `0.0267` | `26.28` | `0.7805` | 2 / 17 | num_opt_steps at max, lr at min, beta at max | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=0.5` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.3884` | `0.0490` | `18.42` | `0.3424` | 1 / 1 | steps at max | `steps=2` |

## pMF-L-16 &mdash; Box inpainting

Degraded observation: LPIPS `0.1010`, PSNR `27.50 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.0807` | `0.0077` | `27.81` | `0.8365` | 5 / 19 | K at min, lr at max | `num_mpc_steps=2` `K=1` `lam=15.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.0669` | `0.0060` | `28.35` | `0.8652` | 6 / 15 | n_ctrl at max | `num_mpc_steps=2` `lam=30.0` `n_ctrl=80` `lr=0.1` |
| PnP-Flow | `0.1166` | `0.0247` | `26.15` | `0.7993` | 0 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2577` | `0.0611` | `25.04` | `0.6885` | 0 / 3 | num_opt_steps at max, lr at min | `steps=2` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.0673` | `0.0155` | `28.52` | `0.8808` | 7 / 17 | num_opt_steps at max, lr at min, beta at max | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=0.2` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.3870` | `0.0505` | `18.47` | `0.3515` | 1 / 1 | steps at min | `steps=1` |

## pMF-L-16 &mdash; Random inpainting

Degraded observation: LPIPS `1.0188`, PSNR `12.72 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.5504` | `0.0305` | `20.06` | `0.3824` | 7 / 19 | K at max, lr at min | `num_mpc_steps=2` `K=3` `lam=15.0` `n_ctrl=20` `lr=0.05` |
| MPC-Delta_t | `0.1323` | `0.0200` | `25.91` | `0.7607` | 2 / 15 | n_ctrl at max | `num_mpc_steps=2` `lam=300.0` `n_ctrl=80` `lr=0.1` |
| PnP-Flow | `0.2540` | `0.0319` | `22.70` | `0.5982` | 0 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2992` | `0.0539` | `24.50` | `0.6462` | 2 / 3 | num_opt_steps at max, lr at min | `steps=2` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.1061` | `0.0161` | `26.40` | `0.7891` | 6 / 17 | num_opt_steps at max, lr at min, beta at max | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=0.5` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6700` | `0.0264` | `13.44` | `0.1130` | 1 / 1 | steps at max | `steps=2` |

## pMF-L-16 &mdash; Stroke painting

Degraded observation: LPIPS `0.4517`, PSNR `19.63 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.4052` | `0.0467` | `18.49` | `0.3473` | 9 / 19 | K at min, lam at max | `num_mpc_steps=2` `K=1` `lam=500.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.4144` | `0.0460` | `18.32` | `0.3348` | 8 / 15 | n_ctrl at min | `num_mpc_steps=2` `lam=15.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.4792` | `0.0431` | `17.52` | `0.3080` | 5 / 11 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.7421` | `0.0368` | `11.72` | `0.1197` | 3 / 3 | num_opt_steps at min, lr at min | `steps=2` `num_opt_steps=40` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.4802` | `0.0377` | `15.68` | `0.2420` | 1 / 17 | num_opt_steps at max, lr at min, mu at max, beta at max | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=2.0` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.4520` | `0.0510` | `17.58` | `0.2964` | 1 / 1 | steps at max | `steps=2` |

---

# SiT-XL/2

SD-VAE latent 4x32x32, standard flow, Torch

## SiT-XL/2 &mdash; Denoising

Degraded observation: LPIPS `0.4712`, PSNR `20.46 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.2844` | `0.0343` | `25.13` | `0.6374` | 2 / 19 | K at min, lam at max, n_ctrl at max, lr at max | `num_mpc_steps=4` `K=1` `lam=500.0` `n_ctrl=40` `lr=0.1` |
| MPC-Delta_t | `0.2335` | `0.0337` | `25.92` | `0.6779` | 1 / 23 | num_mpc_steps at min, n_ctrl at max | `num_mpc_steps=4` `lam=300.0` `n_ctrl=80` `lr=0.1` |
| PnP-Flow | `0.5979` | `0.0457` | `22.23` | `0.5206` | 3 / 29 | alpha at min | `num_pnp_steps=20` `gamma0=0.5` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.6171` | `0.0302` | `19.17` | `0.3602` | 11 / 17 | steps at min, num_opt_steps at max, lr at max | `steps=2` `solver=heun` `num_opt_steps=40` `lr=0.1` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.2239` | `0.0322` | `26.09` | `0.6929` | 5 / 69 | num_opt_steps at max, mu at min | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6717` | `0.0253` | `12.28` | `0.0696` | 2 / 5 | steps at min | `steps=4` `solver=heun` |

## SiT-XL/2 &mdash; Deblurring

Degraded observation: LPIPS `0.2709`, PSNR `25.74 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.2744` | `0.0353` | `25.04` | `0.6599` | 1 / 2 | — | `num_mpc_steps=4` `K=1` `lam=50.0` `n_ctrl=40` `lr=0.1` |
| MPC-Delta_t | `0.2492` | `0.0304` | `25.05` | `0.6694` | 0 / 2 | lam at max | `num_mpc_steps=4` `lam=300.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.6186` | `0.0435` | `21.98` | `0.5066` | 3 / 3 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=0.5` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.5297` | `0.0350` | `20.51` | `0.4031` | 1 / 3 | num_opt_steps at max, lr at min | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.2189` | `0.0328` | `26.05` | `0.7218` | 1 / 11 | num_opt_steps at max, lr at min, mu at min, beta at min | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6082` | `0.0271` | `12.96` | `0.1534` | 0 / 0 | — | `steps=4` `solver=heun` |

## SiT-XL/2 &mdash; Super-resolution

Degraded observation: LPIPS `0.2324`, PSNR `22.73 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.2558` | `0.0369` | `24.71` | `0.6514` | 1 / 2 | lam at max | `num_mpc_steps=4` `K=1` `lam=500.0` `n_ctrl=40` `lr=0.1` |
| MPC-Delta_t | `0.2739` | `0.0337` | `24.26` | `0.6318` | 0 / 2 | lam at max | `num_mpc_steps=4` `lam=300.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.6081` | `0.0432` | `21.58` | `0.4663` | 3 / 3 | gamma0 at min, alpha at min | `num_pnp_steps=20` `gamma0=0.25` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.5433` | `0.0409` | `20.46` | `0.4098` | 1 / 3 | num_opt_steps at max, lr at min | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.1889` | `0.0353` | `25.58` | `0.7161` | 3 / 11 | num_opt_steps at max, lr at max, mu at min, beta at min | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.1` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6237` | `0.0286` | `13.16` | `0.1740` | 0 / 0 | — | `steps=4` `solver=heun` |

## SiT-XL/2 &mdash; Box inpainting

Degraded observation: LPIPS `0.1010`, PSNR `27.50 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.2414` | `0.0366` | `24.92` | `0.6883` | 1 / 2 | lam at max | `num_mpc_steps=4` `K=1` `lam=500.0` `n_ctrl=40` `lr=0.1` |
| MPC-Delta_t | `0.2225` | `0.0336` | `24.92` | `0.7086` | 1 / 2 | lam at max | `num_mpc_steps=4` `lam=300.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.5981` | `0.0432` | `21.93` | `0.5160` | 2 / 3 | gamma0 at max, alpha at min | `num_pnp_steps=20` `gamma0=0.5` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.5339` | `0.0481` | `20.26` | `0.4064` | 1 / 3 | num_opt_steps at max, lr at min | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.1960` | `0.0300` | `25.95` | `0.7516` | 3 / 11 | num_opt_steps at max, lr at min, mu at min, beta at min | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6195` | `0.0277` | `12.98` | `0.1523` | 0 / 0 | — | `steps=4` `solver=heun` |

## SiT-XL/2 &mdash; Random inpainting

Degraded observation: LPIPS `1.0188`, PSNR `12.72 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.3561` | `0.0240` | `23.49` | `0.5695` | 1 / 2 | lam at max | `num_mpc_steps=4` `K=1` `lam=500.0` `n_ctrl=40` `lr=0.1` |
| MPC-Delta_t | `0.3635` | `0.0293` | `23.40` | `0.5745` | 0 / 2 | lam at max | `num_mpc_steps=4` `lam=300.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.6149` | `0.0482` | `21.82` | `0.4776` | 2 / 3 | gamma0 at min, alpha at min | `num_pnp_steps=20` `gamma0=0.25` `alpha=0.5` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.5939` | `0.0350` | `20.37` | `0.3981` | 1 / 3 | num_opt_steps at max, lr at max | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.1` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.2055` | `0.0346` | `25.31` | `0.7105` | 3 / 11 | num_opt_steps at max, lr at max, mu at min, beta at min | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.1` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.7483` | `0.0364` | `10.78` | `0.0191` | 0 / 0 | — | `steps=4` `solver=heun` |

## SiT-XL/2 &mdash; Stroke painting

Degraded observation: LPIPS `0.4517`, PSNR `19.63 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.5846` | `0.0416` | `18.49` | `0.3611` | 2 / 2 | — | `num_mpc_steps=4` `K=1` `lam=50.0` `n_ctrl=40` `lr=0.1` |
| MPC-Delta_t | `0.5426` | `0.0359` | `17.61` | `0.3391` | 2 / 2 | — | `num_mpc_steps=4` `lam=100.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.7195` | `0.0378` | `18.52` | `0.3808` | 3 / 3 | gamma0 at min, alpha at max | `num_pnp_steps=20` `gamma0=0.25` `alpha=1.0` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.6064` | `0.0402` | `16.82` | `0.2927` | 1 / 3 | num_opt_steps at max, lr at min | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.5632` | `0.0358` | `17.70` | `0.3255` | 9 / 11 | num_opt_steps at max, lr at min, mu at min, beta at min | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6981` | `0.0294` | `13.60` | `0.1839` | 0 / 0 | — | `steps=4` `solver=heun` |

---

# iMF-B-2

SD-VAE latent 32x32x4, MeanFlow, JAX

## iMF-B-2 &mdash; Denoising

Degraded observation: LPIPS `0.4712`, PSNR `20.46 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.4261` | `0.0297` | `23.28` | `0.5273` | 3 / 9 | — | `num_mpc_steps=2` `K=1` `lam=50.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.4593` | `0.0288` | `23.10` | `0.5204` | 2 / 7 | lam at max | `num_mpc_steps=2` `lam=1000.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.5175` | `0.0481` | `23.13` | `0.5337` | 0 / 11 | gamma0 at min | `num_pnp_steps=20` `gamma0=0.5` `alpha=1.0` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.5539` | `0.0455` | `21.62` | `0.4654` | 1 / 19 | steps at max, num_opt_steps at max | `steps=2` `num_opt_steps=80` `lr=0.2` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.3320` | `0.0478` | `24.94` | `0.6418` | 4 / 27 | mu at min | `num_rhso_steps=4` `num_opt_steps=10` `lr=0.1` `mu=0.0` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6387` | `0.0205` | `12.52` | `0.0945` | 2 / 2 | steps at max | `steps=4` |

## iMF-B-2 &mdash; Deblurring

Degraded observation: LPIPS `0.2709`, PSNR `25.74 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.3775` | `0.0347` | `23.51` | `0.5658` | 0 / 4 | lam at max | `num_mpc_steps=2` `K=1` `lam=50.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.3795` | `0.0339` | `23.55` | `0.5681` | 1 / 4 | lam at max | `num_mpc_steps=2` `lam=1000.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.5345` | `0.0456` | `22.42` | `0.4796` | 1 / 1 | gamma0 at min | `num_pnp_steps=20` `gamma0=0.25` `alpha=1.0` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.5938` | `0.0474` | `19.57` | `0.3797` | 1 / 1 | lr at min | `steps=2` `num_opt_steps=40` `lr=0.1` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.3547` | `0.0531` | `25.02` | `0.6660` | 1 / 5 | mu at min, beta at min | `num_rhso_steps=4` `num_opt_steps=10` `lr=0.1` `mu=0.0` `beta=0.25` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.5963` | `0.0422` | `14.08` | `0.1738` | 0 / 0 | — | `steps=2` |

## iMF-B-2 &mdash; Super-resolution

Degraded observation: LPIPS `0.2324`, PSNR `22.73 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.3505` | `0.0352` | `23.53` | `0.5774` | 0 / 4 | lam at max | `num_mpc_steps=2` `K=1` `lam=50.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.3630` | `0.0357` | `23.48` | `0.5747` | 0 / 4 | lam at max | `num_mpc_steps=2` `lam=1000.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.5406` | `0.0431` | `21.94` | `0.4693` | 1 / 1 | gamma0 at max | `num_pnp_steps=20` `gamma0=0.5` `alpha=1.0` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.6145` | `0.0483` | `19.26` | `0.3619` | 1 / 1 | lr at min | `steps=2` `num_opt_steps=40` `lr=0.1` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.3194` | `0.0578` | `25.03` | `0.6687` | 1 / 5 | mu at min, beta at min | `num_rhso_steps=4` `num_opt_steps=10` `lr=0.1` `mu=0.0` `beta=0.25` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.5758` | `0.0431` | `14.02` | `0.1934` | 0 / 0 | — | `steps=2` |

## iMF-B-2 &mdash; Box inpainting

Degraded observation: LPIPS `0.1010`, PSNR `27.50 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.3559` | `0.0324` | `23.00` | `0.5784` | 0 / 4 | lam at max | `num_mpc_steps=2` `K=1` `lam=50.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.3776` | `0.0324` | `23.19` | `0.5728` | 1 / 4 | lam at max | `num_mpc_steps=2` `lam=1000.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.5209` | `0.0478` | `22.94` | `0.5312` | 1 / 1 | gamma0 at max | `num_pnp_steps=20` `gamma0=0.5` `alpha=1.0` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.6104` | `0.0403` | `19.03` | `0.3615` | 1 / 1 | lr at min | `steps=2` `num_opt_steps=40` `lr=0.1` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.2948` | `0.0391` | `25.02` | `0.6781` | 1 / 5 | mu at min, beta at min | `num_rhso_steps=4` `num_opt_steps=10` `lr=0.1` `mu=0.0` `beta=0.25` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.5842` | `0.0298` | `13.16` | `0.1455` | 0 / 0 | — | `steps=2` |

## iMF-B-2 &mdash; Random inpainting

Degraded observation: LPIPS `1.0188`, PSNR `12.72 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.5424` | `0.0266` | `21.70` | `0.4260` | 0 / 4 | lam at max | `num_mpc_steps=2` `K=1` `lam=50.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.5805` | `0.0240` | `21.60` | `0.4262` | 1 / 4 | lam at max | `num_mpc_steps=2` `lam=1000.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.5276` | `0.0439` | `22.05` | `0.4689` | 1 / 1 | gamma0 at max | `num_pnp_steps=20` `gamma0=0.5` `alpha=1.0` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.6490` | `0.0253` | `18.98` | `0.3529` | 1 / 1 | lr at max | `steps=2` `num_opt_steps=40` `lr=0.2` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.3391` | `0.0406` | `24.22` | `0.6175` | 1 / 5 | mu at min, beta at min | `num_rhso_steps=4` `num_opt_steps=10` `lr=0.1` `mu=0.0` `beta=0.25` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.7234` | `0.0135` | `11.07` | `0.0193` | 0 / 0 | — | `steps=2` |

## iMF-B-2 &mdash; Stroke painting

Degraded observation: LPIPS `0.4517`, PSNR `19.63 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | ties | edge | best configuration |
|---|---|---|---|---|---|---|---|
| MPC-RHC | `0.5965` | `0.0375` | `17.48` | `0.3232` | 0 / 4 | lam at min | `num_mpc_steps=2` `K=1` `lam=5.0` `n_ctrl=20` `lr=0.1` |
| MPC-Delta_t | `0.6117` | `nan` | `17.14` | `0.3061` | 0 / 4 | lam at min | `num_mpc_steps=2` `lam=15.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.6572` | `0.0429` | `16.93` | `0.2677` | 1 / 1 | gamma0 at min | `num_pnp_steps=20` `gamma0=0.25` `alpha=1.0` `noise_samples=1` `phi_normalization=half_sum_squared` |
| D-Flow | `0.6672` | `0.0561` | `14.98` | `0.2605` | 1 / 1 | lr at min | `steps=2` `num_opt_steps=40` `lr=0.1` `phi_normalization=half_mean_squared_per_measurement` |
| RHSO | `0.6480` | `0.0387` | `16.51` | `0.2862` | 4 / 5 | mu at min, beta at max | `num_rhso_steps=4` `num_opt_steps=10` `lr=0.1` `mu=0.0` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6565` | `0.0372` | `14.26` | `0.2142` | 0 / 0 | — | `steps=2` |

