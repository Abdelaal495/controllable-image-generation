"""One problem, one model, one t0, one epsilon -- compare reconstruction strategies.

A small research repository merged from two Jupyter notebooks:

    src/config.py         registries, validation, sweeps, the run planner
    src/data.py           the shared ImageNet source pool
    src/problems.py       the six inverse problems (the ONE degradation layer)
    src/models/           one adapter per model: JiT, pMF, SiT, iMF
    src/reconstruction.py the (dynamics family, method) -> reconstructor registry
    src/sdedit.py         ordinary reconstruction (the paired baseline)
    src/mpc.py            MPC-RHC and MPC-delta_t
    src/pnp.py            PnP-Flow
    src/dflow.py          D-Flow
    src/memory.py         per-job GPU peak-memory measurement
    src/metrics.py        PSNR, SSIM, LPIPS, measurement consistency
    src/visualization.py  comparison grids and summary plots
    src/checks.py         parity, gradient and fairness tests

The strategies compared are SDEdit (baseline), MPC-RHC, MPC-delta_t, PnP-Flow and D-Flow;
see docs/methods_pnp_dflow.md for what is published and what this repository adapts.
"""

__version__ = "1.0.0"
