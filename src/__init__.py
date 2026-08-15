"""SDEdit vs. MPC-Flow: one problem, one model, one t0, one epsilon -- compare strategies.

A small research repository merged from two Jupyter notebooks:

    src/config.py         registries, validation, sweeps, the run planner
    src/data.py           the shared ImageNet source pool
    src/problems.py       the six inverse problems (the ONE degradation layer)
    src/models/           one adapter per model: JiT, pMF, SiT, iMF
    src/sdedit.py         ordinary reconstruction
    src/mpc.py            MPC-RHC and MPC-delta_t
    src/metrics.py        PSNR, SSIM, LPIPS, measurement consistency
    src/visualization.py  comparison grids and summary plots
    src/checks.py         parity, gradient and fairness tests
"""

__version__ = "1.0.0"
