# Executable tests for PnP-Flow and D-Flow

Two scripts that run the reconstruction code against **synthetic analytic "models"** — no
checkpoint, no GPU, no download. They exist because the structural checks in `src/checks.py`
run inside a real experiment, and these can run in seconds on a laptop or a login node.

```bash
python tests/test_meanflow_pnp_dflow.py     # needs jax + optax
python tests/test_flow_trajectory.py        # needs jax
```

`test_meanflow_pnp_dflow.py` executes `pnp.meanflow_pnp` and `dflow.meanflow_dflow`
end to end against a toy MeanFlow adapter and checks: the `1 + N·M` denoiser accounting and
zero generative backprops; the correction schedule and step sizes; bitwise determinism;
that the reprojection noise does **not** move when `gamma0` does; that the fidelity actually
falls; that Adam reduces the D-Flow objective; that the returned reconstruction is the
trajectory of the **final** `q` rather than a stale iterate; that the gradient with respect
to `q` is finite and non-zero; and that the planner's cost model matches the measured
counters.

`test_flow_trajectory.py` executes the framework-neutral parts of the standard-flow paths:
`dflow._integrate_flow` for euler / heun / rk4 with and without JiT's final-Euler policy,
compared **bitwise** against an independent transcription of the reference integrator in
`sdedit.py`; the evaluation counts; the terminal gradient through a multi-step trajectory
and that a descent step reduces the loss; and the PnP schedule and per-image reprojection
noise.

## What these tests do NOT cover

The toy adapters are not the real models, so nothing here says anything about
reconstruction *quality*. More specifically:

* `pnp.flow_pnp` and `dflow.flow_dflow` are **not** executed — they call
  `torch.autograd.grad`, `torch.optim.Adam` and `torch.no_grad`, which need PyTorch. Their
  trajectory arithmetic and counters are covered indirectly (the integrator above and the
  MeanFlow twins), but the torch glue itself is not.
* `src/memory.py`'s Torch and NVML paths need a CUDA device; on a CPU box the profiler
  reports `cpu_no_gpu_memory` and the columns stay empty.
* The VAE-decoder gradient path for latent models (SiT, iMF) needs a real checkpoint.

Run the real per-model checks for those: `python run.py --config ... --check` executes
`pnp_initial_projection`, `pnp_determinism`, `dflow_gradient` and `dflow_optimisation`
against the actual loaded model, and they are recorded in `checks.json`.
