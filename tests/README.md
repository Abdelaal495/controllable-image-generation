# Executable tests for PnP-Flow and D-Flow

Two scripts that run the reconstruction code against **synthetic analytic "models"** — no
checkpoint, no GPU, no download. They exist because the structural checks in `src/checks.py`
run inside a real experiment, and these can run in seconds on a laptop or a login node.

```bash
python tests/test_meanflow_pnp_dflow.py     # needs jax + optax
python tests/test_flow_trajectory.py        # needs jax
python tests/test_beta_schedule.py          # pure Python (+ pyyaml for the config checks)
python tests/test_rhso.py                   # needs jax + optax
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

`test_beta_schedule.py` needs nothing but the standard library and checks the universal
power-law grid: `beta` validation, exact endpoints, strict monotonicity, the worked
`beta = 0.5` / `beta = 2` examples, interval direction, PnP's times staying strictly inside
`(s0, 1)` for every `beta`, that `MPC-Δt`'s inverse-delta scaling uses the **local** `dt_k`,
and — most importantly — that `beta = 1` reproduces the pre-`beta` grid **bitwise**, against
an independent transcription of the original expressions.

`test_rhso.py` executes `rhso.meanflow_rhso` against toy MeanFlow adapters (one pixel-space,
one latent with a differentiable decoder) and asserts the *structure* rather than the
output: the recorded transition trace shows `M` direct `s_k → 1` planning calls and exactly
one `s_k → s_{k+1}` execution per stage; `lr = 0` degenerates the run to plain successive
execution, which is what proves `q` starts from `x_k`; the Adam state is verified fresh at
every stage against a reference (with the carried-moments variant shown to differ); the
counters match `N·M` and `N·(M+1)` and the planner's cost model; the result is deterministic
and distinct from D-Flow. The standard-flow parts that are not torch-specific — the planning
suffix, the evaluation counts, `flow_step`, and `integrate_flow` over a non-uniform suffix
compared bitwise against a reference integrator — are executed too.

It also covers RHSO's optional state-anchor regularisation `mu`: that `R` is zero at the
anchor and positive after a displacement, that its gradient equals `(q − anchor)/d` and a
descent step strictly shortens the distance to the anchor, that `R` is summed over the batch
and normalised by the state dimensionality, that the anchor stays bitwise fixed across a
stage's inner iterations while the displacement grows from exactly zero, that stage `k+1`
anchors bitwise to stage `k`'s **executed** state (not `x0`, not the optimised `q`), that a
larger `mu` really does hold `q` nearer the anchor, that every compute counter is identical
at `mu = 0` and `mu > 0`, and that a spec carrying `mu = 0.0` and one with no `mu` attribute
at all produce a **bitwise identical** result. Torch/JAX parity is asserted structurally —
both loops are shown (by scanning their bytecode, nested closures included) to call the one
backend-generic `rhso.state_anchor_penalty`, whose NumPy and JAX evaluations agree — because
`flow_rhso` itself cannot run here.

`tests/spec_support.py` builds every spec through the **real** validator and planner: it
assembles a small in-memory configuration, runs `validate_config` and `resolve_run_plan`,
and splits overrides automatically between the configuration (where they are validated) and
`dataclasses.replace`. A test that runs at all has therefore already proved its method and
fields are properly declared. Both scripts also exercise the configuration directly — `beta`
sweeping into distinct job ids and directories, `delta` becoming null at `beta != 1`, RHSO
rejecting `lam`/`K`/`control_cost_normalization`/non-Adam optimisers, `mu` sweeping into
distinct job ids and output paths while leaving the cost estimate untouched, `mu` being
refused for every non-RHSO method, and the warm-up key separating beta- and mu-distinct
jobs. A documented stand-in remains as a fallback if
`src/config.py` cannot be imported at all; each script prints which path it used.

## What these tests do NOT cover

The toy adapters are not the real models, so nothing here says anything about
reconstruction *quality*. More specifically:

* `pnp.flow_pnp`, `dflow.flow_dflow` and `rhso.flow_rhso` are **not** executed — they call
  `torch.autograd.grad`, `torch.optim.Adam` and `torch.no_grad`, which need PyTorch. Their
  trajectory arithmetic and counters are covered indirectly (the integrator above and the
  MeanFlow twins), but the torch glue itself is not.
* `src/memory.py`'s Torch and NVML paths need a CUDA device; on a CPU box the profiler
  reports `cpu_no_gpu_memory` and the columns stay empty.
* The VAE-decoder gradient path for latent models (SiT, iMF) needs a real checkpoint.

Run the real per-model checks for those: `python run.py --config ... --check` executes
`pnp_initial_projection`, `pnp_determinism`, `dflow_gradient` and `dflow_optimisation`
against the actual loaded model, and they are recorded in `checks.json`.



