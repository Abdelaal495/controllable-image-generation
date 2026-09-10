# Final benchmark on the frozen ImageNet-100 (class seed 42 / image seed 43), t0 = 1.0, 100 images per cell

Generated 2026-09-10 by `scripts/make_final_tables.py` from outputs/final_jit, outputs/final_pmf.

Each cell runs the Stage-2 winner of its (model, problem, method); `+/- se` is the standard error of LPIPS over the 100 images. Runtimes were measured with several processes sharing each A100 and are inflated by contention; treat them as upper bounds.

## LPIPS summary (lower is better)

| model | problem | SDEdit | PnP-Flow | D-Flow | MPC-RHC | MPC-Delta_t | RHSO | best |
|---|---|---|---|---|---|---|---|---|
| JiT-B/16 | Denoising | 0.7202 | **0.1070** | 0.2953 | 0.1255 | 0.1933 | 0.1545 | PnP-Flow |
| JiT-B/16 | Deblurring | 0.7202 | 0.1518 | 0.2754 | 0.1817 | 0.1399 | **0.0868** | RHSO |
| JiT-B/16 | 2x SR | 0.7202 | **0.0846** | 0.2854 | 0.1497 | 0.0903 | 0.0933 | PnP-Flow |
| JiT-B/16 | Random inpainting | 0.7202 | 0.0874 | 0.2849 | 0.1645 | **0.0745** | 0.0752 | MPC-Delta_t |
| JiT-B/16 | Box inpainting | 0.7202 | 0.0509 | 0.1563 | 0.0874 | 0.0517 | **0.0420** | RHSO |
| pMF-L/16 | Denoising | 0.7189 | **0.0809** | 0.1988 | 0.1226 | 0.1665 | 0.0935 | PnP-Flow |
| pMF-L/16 | Deblurring | 0.7189 | 0.1652 | 0.1769 | 0.1593 | 0.1535 | **0.1035** | RHSO |
| pMF-L/16 | 2x SR | 0.7189 | 0.1202 | 0.1478 | 0.2261 | **0.0874** | 0.0947 | MPC-Delta_t |
| pMF-L/16 | Random inpainting | 0.7189 | 0.1166 | 0.1487 | 0.2365 | **0.0774** | 0.0784 | MPC-Delta_t |
| pMF-L/16 | Box inpainting | 0.7189 | 0.0572 | 0.1891 | 0.0956 | 0.0590 | **0.0462** | RHSO |

## JiT-B/16

### Denoising

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.4857 | - | 20.51 | 0.3926 | - | - | |
| SDEdit | 0.7202 | 0.0079 | 9.17 | 0.0136 | - | 0.6 | steps=25 |
| PnP-Flow | 0.1070 | 0.0052 | 28.99 | 0.8015 | - | 0.7 | num_pnp_steps=50 gamma0=200000.0 alpha=0.5 |
| D-Flow | 0.2953 | 0.0111 | 26.40 | 0.6846 | - | 28.5 | num_opt_steps=2560 lr=0.3 |
| MPC-RHC | 0.1255 | 0.0060 | 29.14 | 0.8073 | - | 5.2 | K=2 lam=15.0 lr=0.05 |
| MPC-Delta_t | 0.1933 | 0.0077 | 27.98 | 0.7264 | - | 4.1 | num_mpc_steps=8 n_ctrl=20 lam=30.0 lr=0.3 |
| RHSO | 0.1545 | 0.0070 | 28.30 | 0.8035 | - | 21.7 | num_rhso_steps=4 num_opt_steps=40 lr=0.005 mu=0.2 |

### Deblurring

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2729 | - | 25.91 | 0.7120 | - | - | |
| SDEdit | 0.7202 | 0.0079 | 9.17 | 0.0136 | - | 0.4 | steps=25 |
| PnP-Flow | 0.1518 | 0.0072 | 29.00 | 0.8528 | - | 2.2 | num_pnp_steps=200 gamma0=1200000.0 alpha=0.5 |
| D-Flow | 0.2754 | 0.0090 | 26.92 | 0.7388 | - | 77.6 | num_opt_steps=5120 lr=0.3 |
| MPC-RHC | 0.1817 | 0.0064 | 29.15 | 0.8097 | - | 0.1 | K=1 lam=60.0 lr=0.05 |
| MPC-Delta_t | 0.1399 | 0.0065 | 29.51 | 0.8141 | - | 46.3 | num_mpc_steps=8 n_ctrl=320 lam=1000.0 lr=0.1 |
| RHSO | 0.0868 | 0.0048 | 30.32 | 0.8743 | - | 27.9 | num_rhso_steps=8 num_opt_steps=20 lr=0.01 mu=0.0 |

### 2x SR

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2262 | - | 22.81 | 0.6491 | - | - | |
| SDEdit | 0.7202 | 0.0079 | 9.17 | 0.0136 | - | 0.5 | steps=25 |
| PnP-Flow | 0.0846 | 0.0040 | 27.51 | 0.8231 | - | 0.3 | num_pnp_steps=20 gamma0=200000.0 alpha=0.25 |
| D-Flow | 0.2854 | 0.0081 | 25.08 | 0.6794 | - | 68.6 | num_opt_steps=2560 lr=0.3 |
| MPC-RHC | 0.1497 | 0.0060 | 26.76 | 0.7648 | - | 4.5 | K=2 lam=720.0 lr=0.05 |
| MPC-Delta_t | 0.0903 | 0.0044 | 27.35 | 0.8035 | - | 7.2 | num_mpc_steps=8 n_ctrl=40 lam=540.0 lr=0.05 |
| RHSO | 0.0933 | 0.0056 | 27.80 | 0.8293 | - | 19.1 | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0.2 |

### Random inpainting

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 1.0438 | - | 12.31 | 0.1372 | - | - | |
| SDEdit | 0.7202 | 0.0079 | 9.17 | 0.0136 | 8.59 | 0.5 | steps=25 |
| PnP-Flow | 0.0874 | 0.0038 | 27.65 | 0.8270 | 23.74 | 0.7 | num_pnp_steps=50 gamma0=100000.0 alpha=0.1 |
| D-Flow | 0.2849 | 0.0078 | 25.07 | 0.6880 | 22.20 | 74.7 | num_opt_steps=2560 lr=0.3 |
| MPC-RHC | 0.1645 | 0.0069 | 26.56 | 0.7528 | 23.79 | 4.7 | K=2 lam=240.0 lr=0.02 |
| MPC-Delta_t | 0.0745 | 0.0039 | 28.27 | 0.8526 | 24.36 | 3.8 | num_mpc_steps=8 n_ctrl=20 lam=1000.0 lr=0.05 |
| RHSO | 0.0752 | 0.0045 | 28.36 | 0.8533 | 24.20 | 20.2 | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0.0 |

### Box inpainting

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.1202 | - | 26.20 | 0.8098 | - | - | |
| SDEdit | 0.7202 | 0.0079 | 9.17 | 0.0136 | 8.90 | 0.6 | steps=25 |
| PnP-Flow | 0.0509 | 0.0027 | 31.28 | 0.9007 | 17.13 | 0.3 | num_pnp_steps=50 gamma0=200000.0 alpha=0.1 |
| D-Flow | 0.1563 | 0.0067 | 23.79 | 0.8400 | 8.45 | 211.7 | num_opt_steps=20480 lr=0.3 |
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
| D-Flow | 0.1988 | 0.0110 | 27.80 | 0.7434 | - | 20.9 | num_opt_steps=1280 lr=0.1 |
| MPC-RHC | 0.1226 | 0.0117 | 28.89 | 0.8055 | - | 15.6 | K=2 lam=30.0 lr=0.1 |
| MPC-Delta_t | 0.1665 | 0.0064 | 28.29 | 0.7442 | - | 2.1 | num_mpc_steps=8 n_ctrl=20 lam=30.0 lr=0.15 |
| RHSO | 0.0935 | 0.0072 | 29.45 | 0.8258 | - | 8.1 | num_rhso_steps=4 num_opt_steps=40 lr=0.03 mu=1.0 |

### Deblurring

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2729 | - | 25.91 | 0.7120 | - | - | |
| SDEdit | 0.7189 | 0.0074 | 9.11 | 0.0194 | - | 0.0 | steps=2 |
| PnP-Flow | 0.1652 | 0.0095 | 28.20 | 0.8187 | - | 2.0 | num_pnp_steps=200 gamma0=5000000.0 alpha=1.0 |
| D-Flow | 0.1769 | 0.0194 | 28.07 | 0.7956 | - | 178.8 | num_opt_steps=5120 lr=0.03 |
| MPC-RHC | 0.1593 | 0.0062 | 28.95 | 0.8042 | - | 1.0 | K=1 lam=60.0 lr=0.2 |
| MPC-Delta_t | 0.1535 | 0.0086 | 29.26 | 0.8065 | - | 5.2 | num_mpc_steps=8 n_ctrl=40 lam=1000.0 lr=0.1 |
| RHSO | 0.1035 | 0.0081 | 30.24 | 0.8724 | - | 3.2 | num_rhso_steps=8 num_opt_steps=20 lr=0.01 mu=0.0 |

### 2x SR

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 0.2262 | - | 22.81 | 0.6491 | - | - | |
| SDEdit | 0.7189 | 0.0074 | 9.11 | 0.0194 | - | 0.0 | steps=2 |
| PnP-Flow | 0.1202 | 0.0065 | 26.34 | 0.7883 | - | 0.2 | num_pnp_steps=20 gamma0=100000.0 alpha=0.1 |
| D-Flow | 0.1478 | 0.0129 | 25.87 | 0.7719 | - | 42.0 | num_opt_steps=2560 lr=0.03 |
| MPC-RHC | 0.2261 | 0.0103 | 25.09 | 0.7068 | - | 18.5 | K=3 lam=180.0 lr=0.05 |
| MPC-Delta_t | 0.0874 | 0.0041 | 27.35 | 0.8064 | - | 4.0 | num_mpc_steps=8 n_ctrl=40 lam=360.0 lr=0.15 |
| RHSO | 0.0947 | 0.0060 | 27.50 | 0.8210 | - | 7.4 | num_rhso_steps=4 num_opt_steps=40 lr=0.01 mu=0.2 |

### Random inpainting

| method | LPIPS | +/- se | PSNR | SSIM | missing PSNR | s/image | hyperparameters |
|---|---|---|---|---|---|---|---|
| degraded input | 1.0438 | - | 12.31 | 0.1372 | - | - | |
| SDEdit | 0.7189 | 0.0074 | 9.11 | 0.0194 | 8.63 | 0.0 | steps=2 |
| PnP-Flow | 0.1166 | 0.0054 | 26.40 | 0.7885 | 22.95 | 0.5 | num_pnp_steps=20 gamma0=100000.0 alpha=0.1 |
| D-Flow | 0.1487 | 0.0112 | 26.26 | 0.7759 | 22.28 | 116.7 | num_opt_steps=2560 lr=0.03 |
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

