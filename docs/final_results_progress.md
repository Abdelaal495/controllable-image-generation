# Final benchmark -- progress report

Generated 2026-09-02 12:58 from `outputs/final/`. **This run is still in progress.**

| | |
|---|---|
| jobs finished | **103 / 144** (72%) |
| compute finished | **38.7 h / ~169 h** (23%) |
| failures | 0 |
| by model | JIT 31/36, PMF 36/36, SIT 36/36 |

The two percentages differ because iMF alone is 78% of the compute and is scheduled
last. Job count is 72% done; wall clock is 23% done.

## Run configuration

| setting | value |
|---|---|
| `runtime.seed` | **42** |
| `runtime.replicate` | 0 |
| `t0` / `beta` | 0.8 / 1.0 (every job) |
| images per job | **100** (one per ImageNet class) |
| `batch_size` | 2 for all four models |
| image pool | `cache/data/imagenet_val_100`, built with `--num-classes 100 --seed 0` |

The two seeds are independent. `runtime.seed = 42` fixes the generative noise, the
measurement noise, the inpainting masks and the stroke geometry; the epsilon recipe is
`seed(42, model, "x0", image_id, replicate)`. The pool builder's `--seed 0` fixes only
*which* 100 classes are used, and never enters any reconstruction. The pool's first 32
classes are `src.data.IMAGENET_EXAMPLES` in its original order, so those images keep the
same ids -- and therefore the same epsilon -- as every earlier 8-image run.

At 100 images the median per-cell standard error of LPIPS is **0.0098**, against roughly
0.030-0.048 during tuning at 8 images -- about a 3-4x tightening.

---

## JiT-B/16 (pixel, standard flow) &mdash; 31/36 jobs

### Denoising

Degraded input: LPIPS `0.4472`, PSNR `20.49 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO | `0.1123` | `0.0058` | `29.11` | `0.8174` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=2.0` `beta=0.5` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.1771` | `0.0102` | `26.78` | `0.6610` | `num_mpc_steps=4` `lam=30.0` `n_ctrl=80` `lr=0.1` |
| MPC-RHC | `0.1056` | `0.0040` | `28.42` | `0.8050` | `num_mpc_steps=4` `K=3` `lam=5.0` `n_ctrl=20` `lr=0.05` |
| PnP-Flow **<-- best** | `0.1004` | `0.0039` | `28.95` | `0.8253` | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.1839` | `0.0098` | `28.00` | `0.7867` | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.3877` | `0.0096` | `18.77` | `0.3814` | `steps=25` `solver=heun` |

### Deblurring

Degraded input: LPIPS `0.2704`, PSNR `25.52 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO **<-- best** | `0.1229` | `0.0077` | `29.64` | `0.8675` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.2` `beta=0.5` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.1582` | `0.0073` | `29.25` | `0.8325` | `num_mpc_steps=4` `lam=300.0` `n_ctrl=80` `lr=0.1` |
| MPC-RHC | `0.1631` | `0.0061` | `28.82` | `0.8249` | `num_mpc_steps=4` `K=1` `lam=50.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.3333` | `0.0135` | `26.12` | `0.7218` | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2829` | `0.0124` | `25.95` | `0.7100` | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.1` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.4064` | `0.0102` | `18.88` | `0.3762` | `steps=25` `solver=heun` |

### Super-resolution

Degraded input: LPIPS `0.2235`, PSNR `22.40 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| D-Flow **<-- best** | `0.2084` | `0.0107` | `26.67` | `0.7602` | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |

### Box inpainting

Degraded input: LPIPS `0.1133`, PSNR `26.23 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO **<-- best** | `0.0425` | `0.0022` | `28.16` | `0.9184` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.2` `beta=0.5` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.0656` | `0.0045` | `27.91` | `0.8830` | `num_mpc_steps=4` `lam=100.0` `n_ctrl=40` `lr=0.1` |
| MPC-RHC | `0.0609` | `0.0024` | `27.16` | `0.8846` | `num_mpc_steps=4` `K=3` `lam=50.0` `n_ctrl=20` `lr=0.05` |
| PnP-Flow | `0.1533` | `0.0061` | `27.41` | `0.8281` | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2231` | `0.0112` | `25.57` | `0.7740` | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.3999` | `0.0096` | `18.78` | `0.4006` | `steps=4` `solver=heun` |

### Random inpainting

Degraded input: LPIPS `1.0300`, PSNR `12.39 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO | `0.0941` | `0.0053` | `26.91` | `0.8131` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.5` `beta=0.5` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t **<-- best** | `0.0755` | `0.0049` | `27.20` | `0.8244` | `num_mpc_steps=4` `lam=1000.0` `n_ctrl=40` `lr=0.1` |
| MPC-RHC | `0.1254` | `0.0048` | `26.61` | `0.7790` | `num_mpc_steps=4` `K=3` `lam=150.0` `n_ctrl=20` `lr=0.05` |
| PnP-Flow | `0.3275` | `0.0115` | `24.01` | `0.6199` | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2587` | `0.0106` | `25.66` | `0.7121` | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6443` | `0.0099` | `13.09` | `0.1448` | `steps=25` `solver=heun` |

### Stroke painting

Degraded input: LPIPS `0.4309`, PSNR `19.22 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO **<-- best** | `0.3970` | `0.0113` | `18.10` | `0.3543` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=1.0` `beta=0.5` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.4236` | `0.0137` | `19.39` | `0.4114` | `num_mpc_steps=4` `lam=15.0` `n_ctrl=20` `lr=0.1` |
| MPC-RHC | `0.4162` | `0.0127` | `19.54` | `0.4145` | `num_mpc_steps=4` `K=3` `lam=5.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.5569` | `0.0179` | `19.55` | `0.4302` | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.4922` | `0.0111` | `14.69` | `0.2994` | `steps=2` `solver=heun` `num_opt_steps=40` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.4658` | `0.0116` | `17.88` | `0.3232` | `steps=25` `solver=heun` |

---

## pMF-L-16 (pixel, MeanFlow) &mdash; 36/36 jobs

### Denoising

Degraded input: LPIPS `0.4472`, PSNR `20.49 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO | `0.1027` | `0.0059` | `28.63` | `0.8081` | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=2.0` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.2293` | `0.0086` | `25.40` | `0.6187` | `num_mpc_steps=2` `lam=5.0` `n_ctrl=40` `lr=0.1` |
| MPC-RHC | `0.1206` | `0.0074` | `28.05` | `0.7869` | `num_mpc_steps=2` `K=2` `lam=15.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow **<-- best** | `0.0922` | `0.0032` | `28.18` | `0.8105` | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2220` | `0.0118` | `26.05` | `0.7156` | `steps=2` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.3677` | `0.0099` | `18.72` | `0.3833` | `steps=2` |

### Deblurring

Degraded input: LPIPS `0.2704`, PSNR `25.52 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO | `0.2656` | `0.0146` | `27.62` | `0.7672` | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=0.05` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.1809` | `0.0078` | `28.59` | `0.8035` | `num_mpc_steps=2` `lam=100.0` `n_ctrl=80` `lr=0.1` |
| MPC-RHC **<-- best** | `0.1723` | `0.0070` | `27.59` | `0.7626` | `num_mpc_steps=2` `K=1` `lam=50.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.2092` | `0.0094` | `25.22` | `0.7179` | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2694` | `0.0156` | `25.35` | `0.6917` | `steps=2` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.3830` | `0.0100` | `18.66` | `0.3681` | `steps=2` |

### Super-resolution

Degraded input: LPIPS `0.2235`, PSNR `22.40 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO | `0.1125` | `0.0059` | `26.82` | `0.7976` | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=0.5` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t **<-- best** | `0.1096` | `0.0060` | `26.76` | `0.7814` | `num_mpc_steps=2` `lam=1000.0` `n_ctrl=80` `lr=0.1` |
| MPC-RHC | `0.2249` | `0.0080` | `24.32` | `0.6240` | `num_mpc_steps=2` `K=2` `lam=50.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.2639` | `0.0088` | `22.67` | `0.6075` | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2234` | `0.0112` | `25.14` | `0.7023` | `steps=2` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.3722` | `0.0097` | `18.56` | `0.3705` | `steps=2` |

### Box inpainting

Degraded input: LPIPS `0.1133`, PSNR `26.23 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO **<-- best** | `0.0609` | `0.0032` | `27.64` | `0.9019` | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=0.2` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.0820` | `0.0051` | `27.31` | `0.8673` | `num_mpc_steps=2` `lam=30.0` `n_ctrl=80` `lr=0.1` |
| MPC-RHC | `0.0948` | `0.0060` | `26.87` | `0.8395` | `num_mpc_steps=2` `K=1` `lam=15.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.1070` | `0.0035` | `25.82` | `0.8252` | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2546` | `0.0122` | `24.05` | `0.7032` | `steps=2` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.3715` | `0.0094` | `18.44` | `0.3837` | `steps=1` |

### Random inpainting

Degraded input: LPIPS `1.0300`, PSNR `12.39 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO **<-- best** | `0.0957` | `0.0046` | `27.06` | `0.8095` | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=0.5` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.1113` | `0.0052` | `26.85` | `0.7958` | `num_mpc_steps=2` `lam=300.0` `n_ctrl=80` `lr=0.1` |
| MPC-RHC | `0.4928` | `0.0139` | `19.96` | `0.4114` | `num_mpc_steps=2` `K=3` `lam=15.0` `n_ctrl=20` `lr=0.05` |
| PnP-Flow | `0.2360` | `0.0082` | `23.02` | `0.6316` | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.2432` | `0.0133` | `25.00` | `0.6890` | `steps=2` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6444` | `0.0105` | `13.02` | `0.1311` | `steps=2` |

### Stroke painting

Degraded input: LPIPS `0.4309`, PSNR `19.22 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO | `0.4335` | `0.0107` | `16.46` | `0.2957` | `num_rhso_steps=4` `num_opt_steps=20` `lr=0.05` `mu=2.0` `beta=0.5` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.4098` | `0.0109` | `18.43` | `0.3492` | `num_mpc_steps=2` `lam=15.0` `n_ctrl=20` `lr=0.1` |
| MPC-RHC **<-- best** | `0.3985` | `0.0114` | `18.57` | `0.3548` | `num_mpc_steps=2` `K=1` `lam=500.0` `n_ctrl=20` `lr=0.1` |
| PnP-Flow | `0.4379` | `0.0125` | `17.68` | `0.3401` | `num_pnp_steps=20` `gamma0=1.0` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.6237` | `0.0138` | `12.74` | `0.1801` | `steps=2` `num_opt_steps=40` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.4504` | `0.0116` | `17.61` | `0.3094` | `steps=2` |

---

## SiT-XL/2 (latent, standard flow) &mdash; 36/36 jobs

### Denoising

Degraded input: LPIPS `0.4472`, PSNR `20.49 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO **<-- best** | `0.2003` | `0.0081` | `26.94` | `0.7287` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.2014` | `0.0077` | `26.79` | `0.7180` | `num_mpc_steps=4` `lam=300.0` `n_ctrl=80` `lr=0.1` |
| MPC-RHC | `0.2406` | `0.0077` | `25.92` | `0.6805` | `num_mpc_steps=4` `K=1` `lam=500.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.5675` | `0.0142` | `22.16` | `0.5406` | `num_pnp_steps=20` `gamma0=0.5` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.5953` | `0.0134` | `19.46` | `0.3799` | `steps=2` `solver=heun` `num_opt_steps=40` `lr=0.1` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6624` | `0.0115` | `12.55` | `0.1185` | `steps=4` `solver=heun` |

### Deblurring

Degraded input: LPIPS `0.2704`, PSNR `25.52 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO **<-- best** | `0.1928` | `0.0086` | `26.97` | `0.7553` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.2148` | `0.0084` | `25.91` | `0.7074` | `num_mpc_steps=4` `lam=300.0` `n_ctrl=40` `lr=0.1` |
| MPC-RHC | `0.2378` | `0.0087` | `25.80` | `0.6986` | `num_mpc_steps=4` `K=1` `lam=50.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.5843` | `0.0146` | `22.02` | `0.5290` | `num_pnp_steps=20` `gamma0=0.5` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.4805` | `0.0125` | `20.74` | `0.4267` | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.5756` | `0.0090` | `13.22` | `0.1939` | `steps=4` `solver=heun` |

### Super-resolution

Degraded input: LPIPS `0.2235`, PSNR `22.40 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO **<-- best** | `0.1763` | `0.0086` | `26.17` | `0.7436` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.1` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.2352` | `0.0107` | `24.76` | `0.6583` | `num_mpc_steps=4` `lam=300.0` `n_ctrl=40` `lr=0.1` |
| MPC-RHC | `0.2170` | `0.0096` | `25.29` | `0.6816` | `num_mpc_steps=4` `K=1` `lam=500.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.5666` | `0.0175` | `21.79` | `0.4939` | `num_pnp_steps=20` `gamma0=0.25` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.4803` | `0.0141` | `20.93` | `0.4474` | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.5588` | `0.0092` | `13.34` | `0.2098` | `steps=4` `solver=heun` |

### Box inpainting

Degraded input: LPIPS `0.1133`, PSNR `26.23 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO **<-- best** | `0.1713` | `0.0094` | `26.93` | `0.7814` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.1935` | `0.0088` | `25.88` | `0.7376` | `num_mpc_steps=4` `lam=300.0` `n_ctrl=40` `lr=0.1` |
| MPC-RHC | `0.2060` | `0.0087` | `25.83` | `0.7240` | `num_mpc_steps=4` `K=1` `lam=500.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.5732` | `0.0140` | `21.89` | `0.5367` | `num_pnp_steps=20` `gamma0=0.5` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.4850` | `0.0145` | `20.67` | `0.4395` | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.5788` | `0.0093` | `12.95` | `0.1764` | `steps=4` `solver=heun` |

### Random inpainting

Degraded input: LPIPS `1.0300`, PSNR `12.39 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO **<-- best** | `0.1811` | `0.0100` | `26.08` | `0.7404` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.1` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t | `0.3021` | `0.0096` | `24.03` | `0.6121` | `num_mpc_steps=4` `lam=300.0` `n_ctrl=40` `lr=0.1` |
| MPC-RHC | `0.2962` | `0.0085` | `24.11` | `0.6126` | `num_mpc_steps=4` `K=1` `lam=500.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.5745` | `0.0172` | `21.97` | `0.5040` | `num_pnp_steps=20` `gamma0=0.25` `alpha=0.5` `phi_normalization=half_sum_squared` |
| D-Flow | `0.5471` | `0.0122` | `20.75` | `0.4320` | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.1` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.7529` | `0.0102` | `10.57` | `0.0290` | `steps=4` `solver=heun` |

### Stroke painting

Degraded input: LPIPS `0.4309`, PSNR `19.22 dB`.

| strategy | LPIPS | +/- se | PSNR | SSIM | hyperparameters |
|---|---|---|---|---|---|
| RHSO | `0.5240` | `0.0147` | `17.92` | `0.3601` | `num_rhso_steps=4` `num_opt_steps=40` `lr=0.05` `mu=0.0` `beta=0.25` `solver=heun` `phi_normalization=half_mean_squared_per_measurement` |
| MPC-Delta_t **<-- best** | `0.5192` | `0.0153` | `17.12` | `0.3468` | `num_mpc_steps=4` `lam=100.0` `n_ctrl=40` `lr=0.1` |
| MPC-RHC | `0.5339` | `0.0162` | `18.30` | `0.3810` | `num_mpc_steps=4` `K=1` `lam=50.0` `n_ctrl=40` `lr=0.1` |
| PnP-Flow | `0.6959` | `0.0167` | `18.19` | `0.3898` | `num_pnp_steps=20` `gamma0=0.25` `alpha=1.0` `phi_normalization=half_sum_squared` |
| D-Flow | `0.5415` | `0.0152` | `16.90` | `0.3201` | `steps=2` `solver=heun` `num_opt_steps=80` `lr=0.05` `phi_normalization=half_mean_squared_per_measurement` |
| SDEdit *(baseline)* | `0.6411` | `0.0129` | `13.56` | `0.2107` | `steps=4` `solver=heun` |

