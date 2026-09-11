# Final benchmark on the frozen ImageNet-100 (class seed 42 / image seed 43), t0 = 1.0, 100 images per cell

Generated 2026-09-11 by `scripts/make_final_tables.py` from outputs/final_jit, outputs/final_pmf, outputs/final_sit, outputs/final_imf.

Each cell runs the Stage-2 winner of its (model, problem, method); `+/- se` is the standard error of LPIPS over the 100 images. Runtimes were measured with several processes sharing each A100 and are inflated by contention; treat them as upper bounds.

## LPIPS summary (lower is better)

| model | problem | SDEdit | PnP-Flow | D-Flow | MPC-RHC | MPC-Delta_t | RHSO | best |
|---|---|---|---|---|---|---|---|---|
| JiT-B/16 | Denoising | 0.7202 | **0.1070** | 0.4407 | 0.1255 | 0.1933 | 0.1545 | PnP-Flow |
| JiT-B/16 | Deblurring | 0.7202 | 0.1518 | 0.4795 | 0.1817 | 0.1435 | **0.0868** | RHSO |
| JiT-B/16 | 2x SR | 0.7202 | **0.0846** | 0.4358 | 0.1497 | 0.0903 | 0.0933 | PnP-Flow |
| JiT-B/16 | Random inpainting | 0.7202 | 0.0874 | 0.4501 | 0.1645 | **0.0745** | 0.0752 | MPC-Delta_t |
| JiT-B/16 | Box inpainting | 0.7202 | 0.0509 | 0.4507 | 0.0874 | 0.0517 | **0.0420** | RHSO |
| pMF-L/16 | Denoising | 0.7189 | **0.0809** | 0.2583 | 0.1226 | 0.1665 | 0.0935 | PnP-Flow |
| pMF-L/16 | Deblurring | 0.7189 | 0.1652 | 0.3011 | 0.1593 | 0.1535 | **0.1035** | RHSO |
| pMF-L/16 | 2x SR | 0.7189 | 0.1202 | 0.2107 | 0.2261 | **0.0874** | 0.0947 | MPC-Delta_t |
| pMF-L/16 | Random inpainting | 0.7189 | 0.1166 | 0.2037 | 0.2365 | **0.0774** | 0.0784 | MPC-Delta_t |
| pMF-L/16 | Box inpainting | 0.7189 | 0.0572 | 0.1891 | 0.0956 | 0.0590 | **0.0462** | RHSO |
| SiT-XL/2 | Denoising | 0.7346 | 0.4294 | 0.2835 | 0.2277 | **0.1905** | 0.2013 | MPC-Delta_t |
| SiT-XL/2 | Deblurring | 0.7346 | 0.4573 | 0.3239 | 0.2650 | 0.2165 | **0.2030** | RHSO |
| SiT-XL/2 | 2x SR | 0.7346 | 0.4179 | 0.2838 | 0.2145 | **0.1928** | 0.1934 | MPC-Delta_t |
| SiT-XL/2 | Random inpainting | 0.7346 | 0.4264 | 0.2820 | 0.2187 | 0.1841 | **0.1791** | RHSO |
| SiT-XL/2 | Box inpainting | 0.7346 | 0.4392 | 0.2765 | 0.2232 | 0.1794 | **0.1716** | RHSO |
| iMF-B-2 | Denoising | 0.7309 | 0.3121 | 0.3333 | 0.2202 | **0.1964** | 0.2005 | MPC-Delta_t |
| iMF-B-2 | Deblurring | 0.7309 | 0.3652 | 0.3761 | 0.2649 | 0.2086 | **0.2010** | RHSO |
| iMF-B-2 | 2x SR | 0.7309 | 0.3113 | 0.3226 | 0.2090 | 0.1961 | **0.1889** | RHSO |
| iMF-B-2 | Random inpainting | 0.7309 | 0.3249 | 0.3282 | 0.2131 | 0.1886 | **0.1734** | RHSO |
| iMF-B-2 | Box inpainting | 0.7309 | 0.3199 | 0.3221 | 0.2098 | 0.1821 | **0.1733** | RHSO |

## JiT-B/16

### Denoising

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.4857 | - | 20.51 | 0.3926 | - | - | |
| SDEdit | 0.7202 | 0.0079 | 9.17 | 0.0136 | - | 0.6 | steps=25 |
| PnP-Flow | 0.1070 | 0.0052 | 28.99 | 0.8015 | - | 0.7 | num_pnp_steps=50 gamma0=200000.0 alpha=0.5 |
| D-Flow | 0.4407 | 0.0123 | 23.63 | 0.5688 | - | 5.8 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.1255 | 0.0060 | 29.14 | 0.8073 | - | 5.2 | K=2 lam=15.0 lr=0.05 |
| MPC-Delta_t | 0.1933 | 0.0077 | 27.98 | 0.7264 | - | 4.1 | num_mpc_steps=8 n_ctrl=20 lam=30.0 lr=0.3 |
| RHSO | 0.1545 | 0.0070 | 28.30 | 0.8035 | - | 21.7 | num_rhso_steps=4 num_opt_steps=40 lr=0.005 mu=0.2 |

### Deblurring

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2729 | - | 25.91 | 0.7120 | - | - | |
| SDEdit | 0.7202 | 0.0079 | 9.17 | 0.0136 | - | 0.4 | steps=25 |
| PnP-Flow | 0.1518 | 0.0072 | 29.00 | 0.8528 | - | 2.2 | num_pnp_steps=200 gamma0=1200000.0 alpha=0.5 |
| D-Flow | 0.4795 | 0.0115 | 22.89 | 0.5302 | - | 5.9 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.1817 | 0.0064 | 29.15 | 0.8097 | - | 0.1 | K=1 lam=60.0 lr=0.05 |
| MPC-Delta_t | 0.1435 | 0.0061 | 29.50 | 0.8148 | - | 2.6 | num_mpc_steps=8 n_ctrl=40 lam=1000.0 lr=0.1 |
| RHSO | 0.0868 | 0.0048 | 30.32 | 0.8743 | - | 27.9 | num_rhso_steps=8 num_opt_steps=20 lr=0.01 mu=0.0 |

### 2x SR

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2262 | - | 22.81 | 0.6491 | - | - | |
| SDEdit | 0.7202 | 0.0079 | 9.17 | 0.0136 | - | 0.5 | steps=25 |
| PnP-Flow | 0.0846 | 0.0040 | 27.51 | 0.8231 | - | 0.3 | num_pnp_steps=20 gamma0=200000.0 alpha=0.25 |
| D-Flow | 0.4358 | 0.0125 | 23.30 | 0.5650 | - | 5.8 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.1497 | 0.0060 | 26.76 | 0.7648 | - | 4.5 | K=2 lam=720.0 lr=0.05 |
| MPC-Delta_t | 0.0903 | 0.0044 | 27.35 | 0.8035 | - | 7.2 | num_mpc_steps=8 n_ctrl=40 lam=540.0 lr=0.05 |
| RHSO | 0.0933 | 0.0056 | 27.80 | 0.8293 | - | 19.1 | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0.2 |

### Random inpainting

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 1.0438 | - | 12.31 | 0.1372 | - | - | |
| SDEdit | 0.7202 | 0.0079 | 9.17 | 0.0136 | 8.59 | 0.5 | steps=25 |
| PnP-Flow | 0.0874 | 0.0038 | 27.65 | 0.8270 | 23.74 | 0.7 | num_pnp_steps=50 gamma0=100000.0 alpha=0.1 |
| D-Flow | 0.4501 | 0.0118 | 23.13 | 0.5564 | 21.55 | 5.8 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.1645 | 0.0069 | 26.56 | 0.7528 | 23.79 | 4.7 | K=2 lam=240.0 lr=0.02 |
| MPC-Delta_t | 0.0745 | 0.0039 | 28.27 | 0.8526 | 24.36 | 3.8 | num_mpc_steps=8 n_ctrl=20 lam=1000.0 lr=0.05 |
| RHSO | 0.0752 | 0.0045 | 28.36 | 0.8533 | 24.20 | 20.2 | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0.0 |

### Box inpainting

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.1202 | - | 26.20 | 0.8098 | - | - | |
| SDEdit | 0.7202 | 0.0079 | 9.17 | 0.0136 | 8.90 | 0.6 | steps=25 |
| PnP-Flow | 0.0509 | 0.0027 | 31.28 | 0.9007 | 17.13 | 0.3 | num_pnp_steps=50 gamma0=200000.0 alpha=0.1 |
| D-Flow | 0.4507 | 0.0123 | 21.45 | 0.5716 | 8.99 | 5.8 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.0874 | 0.0051 | 27.44 | 0.8472 | 11.45 | 0.1 | K=1 lam=30.0 lr=0.05 |
| MPC-Delta_t | 0.0517 | 0.0030 | 30.76 | 0.8975 | 15.31 | 1.6 | num_mpc_steps=8 n_ctrl=20 lam=180.0 lr=0.1 |
| RHSO | 0.0420 | 0.0019 | 30.07 | 0.9211 | 15.02 | 28.9 | num_rhso_steps=8 num_opt_steps=20 lr=0.01 mu=0.0 |

## pMF-L/16

### Denoising

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.4857 | - | 20.51 | 0.3926 | - | - | |
| SDEdit | 0.7189 | 0.0074 | 9.11 | 0.0194 | - | 0.0 | steps=2 |
| PnP-Flow | 0.0809 | 0.0037 | 28.29 | 0.8391 | - | 0.1 | num_pnp_steps=10 gamma0=200000.0 alpha=0.1 |
| D-Flow | 0.2583 | 0.0121 | 26.26 | 0.6882 | - | 5.2 | num_opt_steps=320 lr=0.1 |
| MPC-RHC | 0.1226 | 0.0117 | 28.89 | 0.8055 | - | 15.6 | K=2 lam=30.0 lr=0.1 |
| MPC-Delta_t | 0.1665 | 0.0064 | 28.29 | 0.7442 | - | 2.1 | num_mpc_steps=8 n_ctrl=20 lam=30.0 lr=0.15 |
| RHSO | 0.0935 | 0.0072 | 29.45 | 0.8258 | - | 8.1 | num_rhso_steps=4 num_opt_steps=40 lr=0.03 mu=1.0 |

### Deblurring

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2729 | - | 25.91 | 0.7120 | - | - | |
| SDEdit | 0.7189 | 0.0074 | 9.11 | 0.0194 | - | 0.0 | steps=2 |
| PnP-Flow | 0.1652 | 0.0095 | 28.20 | 0.8187 | - | 2.0 | num_pnp_steps=200 gamma0=5000000.0 alpha=1.0 |
| D-Flow | 0.3011 | 0.0189 | 25.58 | 0.6895 | - | 12.7 | num_opt_steps=640 lr=0.03 |
| MPC-RHC | 0.1593 | 0.0062 | 28.95 | 0.8042 | - | 1.0 | K=1 lam=60.0 lr=0.2 |
| MPC-Delta_t | 0.1535 | 0.0086 | 29.26 | 0.8065 | - | 5.2 | num_mpc_steps=8 n_ctrl=40 lam=1000.0 lr=0.1 |
| RHSO | 0.1035 | 0.0081 | 30.24 | 0.8724 | - | 3.2 | num_rhso_steps=8 num_opt_steps=20 lr=0.01 mu=0.0 |

### 2x SR

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2262 | - | 22.81 | 0.6491 | - | - | |
| SDEdit | 0.7189 | 0.0074 | 9.11 | 0.0194 | - | 0.0 | steps=2 |
| PnP-Flow | 0.1202 | 0.0065 | 26.34 | 0.7883 | - | 0.2 | num_pnp_steps=20 gamma0=100000.0 alpha=0.1 |
| D-Flow | 0.2107 | 0.0139 | 25.89 | 0.7320 | - | 10.3 | num_opt_steps=640 lr=0.03 |
| MPC-RHC | 0.2261 | 0.0103 | 25.09 | 0.7068 | - | 18.5 | K=3 lam=180.0 lr=0.05 |
| MPC-Delta_t | 0.0874 | 0.0041 | 27.35 | 0.8064 | - | 4.0 | num_mpc_steps=8 n_ctrl=40 lam=360.0 lr=0.15 |
| RHSO | 0.0947 | 0.0060 | 27.50 | 0.8210 | - | 7.4 | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0.2 |

### Random inpainting

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 1.0438 | - | 12.31 | 0.1372 | - | - | |
| SDEdit | 0.7189 | 0.0074 | 9.11 | 0.0194 | 8.63 | 0.0 | steps=2 |
| PnP-Flow | 0.1166 | 0.0054 | 26.40 | 0.7885 | 22.95 | 0.5 | num_pnp_steps=20 gamma0=100000.0 alpha=0.1 |
| D-Flow | 0.2037 | 0.0118 | 26.22 | 0.7321 | 23.49 | 10.5 | num_opt_steps=640 lr=0.03 |
| MPC-RHC | 0.2365 | 0.0110 | 25.01 | 0.6926 | 22.75 | 19.1 | K=3 lam=180.0 lr=0.05 |
| MPC-Delta_t | 0.0774 | 0.0042 | 28.18 | 0.8427 | 24.01 | 13.3 | num_mpc_steps=8 n_ctrl=40 lam=360.0 lr=0.15 |
| RHSO | 0.0784 | 0.0051 | 28.24 | 0.8480 | 23.98 | 7.6 | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0.0 |

### Box inpainting

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.1202 | - | 26.20 | 0.8098 | - | - | |
| SDEdit | 0.7189 | 0.0074 | 9.11 | 0.0194 | 8.91 | 0.0 | steps=2 |
| PnP-Flow | 0.0572 | 0.0026 | 29.24 | 0.8941 | 13.53 | 0.6 | num_pnp_steps=50 gamma0=200000.0 alpha=0.1 |
| D-Flow | 0.1891 | 0.0099 | 24.58 | 0.7771 | 10.12 | 24.0 | num_opt_steps=640 lr=0.1 |
| MPC-RHC | 0.0956 | 0.0054 | 25.88 | 0.8456 | 9.51 | 0.4 | K=1 lam=30.0 lr=0.05 |
| MPC-Delta_t | 0.0590 | 0.0032 | 29.16 | 0.8927 | 12.87 | 7.4 | num_mpc_steps=8 n_ctrl=40 lam=180.0 lr=0.3 |
| RHSO | 0.0462 | 0.0024 | 29.00 | 0.9155 | 13.79 | 4.7 | num_rhso_steps=8 num_opt_steps=20 lr=0.01 mu=0.0 |

## SiT-XL/2

### Denoising

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.4857 | - | 20.51 | 0.3926 | - | - | |
| SDEdit | 0.7346 | 0.0086 | 8.97 | 0.0182 | - | 0.0 | steps=1 |
| PnP-Flow | 0.4294 | 0.0134 | 25.18 | 0.6800 | - | 4.7 | num_pnp_steps=200 gamma0=100000.0 alpha=0.5 |
| D-Flow | 0.2835 | 0.0106 | 25.42 | 0.6528 | - | 21.3 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.2277 | 0.0109 | 27.15 | 0.7358 | - | 2.4 | K=1 lam=5.0 lr=0.2 |
| MPC-Delta_t | 0.1905 | 0.0099 | 27.58 | 0.7500 | - | 10.0 | num_mpc_steps=8 n_ctrl=40 lam=180.0 lr=0.3 |
| RHSO | 0.2013 | 0.0097 | 27.64 | 0.7461 | - | 129.0 | num_rhso_steps=8 num_opt_steps=80 lr=0.03 mu=0.0 |

### Deblurring

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2729 | - | 25.91 | 0.7120 | - | - | |
| SDEdit | 0.7346 | 0.0086 | 8.97 | 0.0182 | - | 0.0 | steps=1 |
| PnP-Flow | 0.4573 | 0.0136 | 24.83 | 0.6656 | - | 4.8 | num_pnp_steps=200 gamma0=100000.0 alpha=0.5 |
| D-Flow | 0.3239 | 0.0125 | 24.50 | 0.6141 | - | 21.3 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.2650 | 0.0124 | 27.05 | 0.7544 | - | 2.4 | K=1 lam=720.0 lr=0.2 |
| MPC-Delta_t | 0.2165 | 0.0098 | 26.98 | 0.7475 | - | 10.0 | num_mpc_steps=8 n_ctrl=40 lam=360.0 lr=0.3 |
| RHSO | 0.2030 | 0.0109 | 27.61 | 0.7752 | - | 127.6 | num_rhso_steps=8 num_opt_steps=80 lr=0.03 mu=0.0 |

### 2x SR

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2262 | - | 22.81 | 0.6491 | - | - | |
| SDEdit | 0.7346 | 0.0086 | 8.97 | 0.0182 | - | 0.0 | steps=1 |
| PnP-Flow | 0.4179 | 0.0145 | 25.31 | 0.6757 | - | 4.7 | num_pnp_steps=200 gamma0=200000.0 alpha=1.0 |
| D-Flow | 0.2838 | 0.0110 | 24.66 | 0.6408 | - | 21.3 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.2145 | 0.0109 | 26.72 | 0.7487 | - | 2.4 | K=1 lam=720.0 lr=0.2 |
| MPC-Delta_t | 0.1928 | 0.0099 | 26.43 | 0.7407 | - | 4.6 | num_mpc_steps=4 n_ctrl=40 lam=540.0 lr=0.3 |
| RHSO | 0.1934 | 0.0098 | 25.96 | 0.7272 | - | 128.4 | num_rhso_steps=8 num_opt_steps=80 lr=0.03 mu=0.0 |

### Random inpainting

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 1.0438 | - | 12.31 | 0.1372 | - | - | |
| SDEdit | 0.7346 | 0.0086 | 8.97 | 0.0182 | 8.50 | 0.0 | steps=1 |
| PnP-Flow | 0.4264 | 0.0143 | 25.06 | 0.6676 | 23.32 | 4.7 | num_pnp_steps=200 gamma0=200000.0 alpha=1.0 |
| D-Flow | 0.2820 | 0.0102 | 24.65 | 0.6426 | 22.67 | 21.4 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.2187 | 0.0109 | 26.61 | 0.7417 | 23.93 | 2.4 | K=1 lam=60.0 lr=0.2 |
| MPC-Delta_t | 0.1841 | 0.0097 | 26.32 | 0.7389 | 23.38 | 10.0 | num_mpc_steps=8 n_ctrl=40 lam=1000.0 lr=0.3 |
| RHSO | 0.1791 | 0.0096 | 26.35 | 0.7462 | 23.21 | 127.5 | num_rhso_steps=8 num_opt_steps=80 lr=0.03 mu=0.0 |

### Box inpainting

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.1202 | - | 26.20 | 0.8098 | - | - | |
| SDEdit | 0.7346 | 0.0086 | 8.97 | 0.0182 | 9.21 | 0.0 | steps=1 |
| PnP-Flow | 0.4392 | 0.0137 | 24.30 | 0.6750 | 14.75 | 4.7 | num_pnp_steps=200 gamma0=200000.0 alpha=0.75 |
| D-Flow | 0.2765 | 0.0118 | 24.51 | 0.6677 | 13.44 | 21.4 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.2232 | 0.0116 | 26.76 | 0.7638 | 15.91 | 2.4 | K=1 lam=15.0 lr=0.2 |
| MPC-Delta_t | 0.1794 | 0.0104 | 27.16 | 0.7866 | 15.43 | 10.0 | num_mpc_steps=8 n_ctrl=40 lam=360.0 lr=0.3 |
| RHSO | 0.1716 | 0.0103 | 27.05 | 0.8004 | 14.63 | 130.3 | num_rhso_steps=8 num_opt_steps=80 lr=0.03 mu=0.0 |

## iMF-B-2

### Denoising

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.4857 | - | 20.51 | 0.3926 | - | - | |
| SDEdit | 0.7309 | 0.0075 | 8.30 | -0.0052 | - | 0.0 | steps=8 |
| PnP-Flow | 0.3121 | 0.0122 | 26.20 | 0.6902 | - | 5.6 | num_pnp_steps=200 gamma0=100000.0 alpha=1.0 |
| D-Flow | 0.3333 | 0.0138 | 25.97 | 0.6877 | - | 20.4 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.2202 | 0.0107 | 27.44 | 0.7471 | - | 4.1 | K=1 lam=15.0 lr=0.2 |
| MPC-Delta_t | 0.1964 | 0.0098 | 27.77 | 0.7570 | - | 9.6 | num_mpc_steps=8 n_ctrl=40 lam=360.0 lr=0.3 |
| RHSO | 0.2005 | 0.0094 | 27.69 | 0.7473 | - | 20.4 | num_rhso_steps=8 num_opt_steps=80 lr=0.03 mu=0.0 |

### Deblurring

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2729 | - | 25.91 | 0.7120 | - | - | |
| SDEdit | 0.7309 | 0.0075 | 8.30 | -0.0052 | - | 0.0 | steps=8 |
| PnP-Flow | 0.3652 | 0.0117 | 25.41 | 0.6591 | - | 6.9 | num_pnp_steps=200 gamma0=100000.0 alpha=1.0 |
| D-Flow | 0.3761 | 0.0124 | 25.39 | 0.6818 | - | 24.2 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.2649 | 0.0118 | 27.14 | 0.7574 | - | 5.1 | K=1 lam=180.0 lr=0.2 |
| MPC-Delta_t | 0.2086 | 0.0105 | 27.71 | 0.7779 | - | 12.1 | num_mpc_steps=8 n_ctrl=40 lam=1000.0 lr=0.3 |
| RHSO | 0.2010 | 0.0107 | 27.54 | 0.7746 | - | 24.6 | num_rhso_steps=8 num_opt_steps=80 lr=0.03 mu=0.0 |

### 2x SR

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2262 | - | 22.81 | 0.6491 | - | - | |
| SDEdit | 0.7309 | 0.0075 | 8.30 | -0.0052 | - | 0.0 | steps=8 |
| PnP-Flow | 0.3113 | 0.0117 | 25.86 | 0.6868 | - | 5.6 | num_pnp_steps=200 gamma0=100000.0 alpha=1.0 |
| D-Flow | 0.3226 | 0.0135 | 25.30 | 0.6828 | - | 20.2 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.2090 | 0.0106 | 26.67 | 0.7489 | - | 4.1 | K=1 lam=240.0 lr=0.2 |
| MPC-Delta_t | 0.1961 | 0.0097 | 26.20 | 0.7388 | - | 9.5 | num_mpc_steps=8 n_ctrl=40 lam=1000.0 lr=0.3 |
| RHSO | 0.1889 | 0.0095 | 25.83 | 0.7313 | - | 20.2 | num_rhso_steps=8 num_opt_steps=80 lr=0.03 mu=0.0 |

### Random inpainting

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 1.0438 | - | 12.31 | 0.1372 | - | - | |
| SDEdit | 0.7309 | 0.0075 | 8.30 | -0.0052 | 7.95 | 0.0 | steps=8 |
| PnP-Flow | 0.3249 | 0.0120 | 25.60 | 0.6775 | 23.64 | 5.7 | num_pnp_steps=200 gamma0=100000.0 alpha=1.0 |
| D-Flow | 0.3282 | 0.0131 | 25.09 | 0.6775 | 22.87 | 20.3 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.2131 | 0.0111 | 26.55 | 0.7467 | 23.78 | 4.2 | K=1 lam=180.0 lr=0.2 |
| MPC-Delta_t | 0.1886 | 0.0101 | 26.50 | 0.7507 | 23.48 | 9.6 | num_mpc_steps=8 n_ctrl=40 lam=1000.0 lr=0.3 |
| RHSO | 0.1734 | 0.0094 | 26.42 | 0.7524 | 23.28 | 20.6 | num_rhso_steps=8 num_opt_steps=80 lr=0.03 mu=0.0 |

### Box inpainting

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.1202 | - | 26.20 | 0.8098 | - | - | |
| SDEdit | 0.7309 | 0.0075 | 8.30 | -0.0052 | 8.43 | 0.0 | steps=8 |
| PnP-Flow | 0.3199 | 0.0123 | 25.32 | 0.6860 | 15.20 | 5.6 | num_pnp_steps=200 gamma0=100000.0 alpha=1.0 |
| D-Flow | 0.3221 | 0.0141 | 25.13 | 0.7053 | 13.93 | 20.4 | num_opt_steps=640 lr=0.3 |
| MPC-RHC | 0.2098 | 0.0117 | 26.89 | 0.7761 | 15.29 | 4.1 | K=1 lam=60.0 lr=0.2 |
| MPC-Delta_t | 0.1821 | 0.0104 | 27.45 | 0.7970 | 15.49 | 9.6 | num_mpc_steps=8 n_ctrl=40 lam=540.0 lr=0.3 |
| RHSO | 0.1733 | 0.0106 | 27.36 | 0.8036 | 14.89 | 20.3 | num_rhso_steps=8 num_opt_steps=80 lr=0.03 mu=0.0 |

