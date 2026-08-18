# Adding a strategy, a model, or a task

The repository is organised so that these three axes are independent:

```
        a problem instance  ×  a generative model  ×  a reconstruction strategy
             problems.py            models/          sdedit.py, mpc.py, pnp.py, dflow.py,
                                                     rhso.py
```

Anything you add on one axis works with everything already present on the other two,
because they only ever meet through two abstractions: the **inverse-problem instance** and
the **model adapter**. Keep that boundary and the paired-comparison guarantees hold for free.

---

## Adding a reconstruction strategy

This is the common case: a new way of turning `z_t0` into a reconstruction.

**1. Write the function.** Put it in its own module (`src/dps.py`, `src/repaint.py`, …) or
alongside the existing ones. The signature is fixed:

```python
def my_strategy(adapter, cond, x0, problem, spec) -> Tuple[Any, ReconstructionStats]:
    """
    adapter  model adapter: velocity() for standard flows, transition() for MeanFlows
    cond     Conditioning (class labels + guidance); identical across strategies
    x0       the corrupted initial state, already built and SHARED with every other
             strategy at the same t0 — do not rebuild it
    problem  the InverseProblem: y, A(), the mask, the guide
    spec     the resolved JobSpec: t0, steps, and your own hyperparameters
    returns  (final native state, ReconstructionStats)
    """
```

Three rules that keep comparisons fair:

* **Never re-draw noise or rebuild `x0`.** It arrives already built from the shared
  epsilon. Drawing your own would silently unpair your method from every baseline.
* **Never touch NumPy inside a differentiable objective.** Use `make_phi(problem, backend,
  normalization)` and `adapter.to_pixels(state, differentiable=True)`. That keeps one
  measurement operator serving both frameworks and both state spaces.
* **Never write your own time grid.** Call
  `schedule.canonical_time_grid(spec.canonical_start_time, steps, schedule.spec_beta(spec))`
  (or `interior_time_grid` if, like PnP, you need times strictly inside the interval). That
  is the single implementation of the universal power-law schedule
  `s_k = s0 + (1 − s0)(k/N)^beta`; duplicating `(k/N)^beta` locally is how a method silently
  stops honouring `beta`. Use the actual `dt_k = s_{k+1} − s_k` everywhere a step size is
  needed — `t0/N` is only the real step size when `beta = 1`. See
  [`schedule_and_rhso.md`](schedule_and_rhso.md).

Fill in `ReconstructionStats` honestly — `model_evals_total`, `network_forwards`,
`backprops_through_model`, `control_iterations`, and, where they apply, `objective_evals`,
`data_gradient_evals`, `optimizer_iterations` and `denoiser_samples`. The compute-cost
columns are only as good as these counts. Two distinctions the existing methods rely on:

* a gradient of the **data-fidelity term** with respect to the current state is
  `data_gradient_evals`, even when it differentiates a VAE decoder — it is not a
  `backprops_through_model`, which counts backward passes through *generative* evaluations;
* an **optimiser iteration** is not a network evaluation. One D-Flow Adam step traverses a
  whole trajectory, so `optimizer_iterations` and `model_evals_total` differ by design.

**2. Declare it** in `src/config.py`:

```python
METHOD_DECLARATIONS["my_strategy"] = MethodDeclaration(
    "my_strategy", "My Strategy", is_mpc=False, uses_K=False,
    fields=SHARED_FIELDS + ("my_param", "steps"),
    description="One sentence on what it does and what it optimises.")
```

Add any new sweepable hyperparameter to `SWEEPABLE_FIELDS`, give it a validation rule in
`_validate_sweep_values` (bounds, type, and a warning if it invalidates a tuned default),
and a fallback in `BUILTIN_DEFAULTS`. The validator is a **closed set**: an undeclared field
is an error, which is what stops a typo from silently becoming a no-op.

**3. Register it** in `src/reconstruction.py` — the neutral registry that owns the global
`(dynamics family, method) -> reconstructor` dispatch. Add the import and the two lines to
`_build_registry`:

```python
from .my_strategy import my_strategy_flow, my_strategy_meanflow

register_reconstructor(STANDARD_FLOW, "my_strategy", my_strategy_flow)
register_reconstructor(MEANFLOW,      "my_strategy", my_strategy_meanflow)
```

`rhso.py` is the most recent worked example of all six steps end to end.

(Dispatch used to live in `src/mpc.py`, which made that module the owner of unrelated
methods. `mpc.py` now declares only MPC's own entries; `src.mpc.select_reconstructor`
survives as a thin alias so old imports keep working.)

If your method is family-agnostic, register the same function for both. If it only makes
sense for one family, register only that one and list the supported methods in the relevant
`ModelCapabilities` entry — the validator will then reject the invalid combination with a
clear message rather than failing at run time.

**4. Estimate its cost** in `_estimate_cost` so `--dry-run` reports a useful workload. It
receives the whole resolved `values` dict, so a method with its own step semantics returns
its own counts; anything you leave at zero shows up as zero in the plan rather than as a
wrong guess.

**5. Give it a figure label** in `JobSpec.figure_title` and
`visualization.config_label`/`group_label`, and a colour in `CONFIG_COLORS`.

**6. Add it to the warm-up key** in `run.py` (`warmup_key`) if it introduces a field that
changes the traced computation or a compiled shape — otherwise a JAX model will compile
inside your measured region and the runtime column will be wrong.

**7. Check the pairing.** Run with your method plus `sdedit` at the same `t0` and confirm
the `shared_initial_state` check passes — it asserts that jobs differing only by method
start from a bit-identical `z_t0`.

Then it is usable immediately:

```yaml
models:
  jit:
    methods:
      my_strategy: {t0: [0.5, 0.8], my_param: [1, 10]}
```

---

## Adding a generative model

Subclass `StandardFlowAdapter` (instantaneous velocity) or `MeanFlowAdapter` (learned
finite-interval transition) in `src/models/<name>.py`, and implement:

| Method | Purpose |
|---|---|
| `to_native_noise(noise)` | canonical NumPy `N(0,I)` → your array type and layout |
| `encode_pixels(pixels)` | canonical `(N,256,256,3)` guide → native state |
| `to_pixels(state, differentiable)` | native → canonical BHWC; **the differentiable path must keep the graph** |
| `_lerp(guide, noise, keep, add)` | `keep·guide + add·noise` in your array library |
| `velocity(state, s, cond)` *or* `transition(state, s_from, s_to, cond)` | dynamics on the **canonical** clock |

Then add a `MODEL_REGISTRY_DEFAULTS` entry, a `MODEL_CAPABILITIES` entry, a
`@register_adapter` factory, a `@register_prefetch` download-only hook (so clusters can
stage it offline), and a line in `models/__init__.load_adapters`.

Three things that are easy to get wrong:

* **The canonical clock is `s: 0 = noise → 1 = data`.** Convert to your checkpoint's native
  convention inside `velocity`/`transition` and nowhere else. Experiment code must never
  contain a model-specific clock inversion.
* **`to_pixels(differentiable=True)` is load-bearing.** It is what lets one pixel-space
  measurement loss serve latent and pixel models in both frameworks. For latent models the
  gradient must flow through the decoder.
* **Implement `sanity_checks()`.** Return a dict of named callables; they run right after the
  checkpoint loads, so a convention error surfaces in seconds rather than as a bad number
  after an hour.

---

## Adding an inverse problem

In `src/problems.py`, add a `ProblemDeclaration` (required and optional parameters, one
guide mode, defaults) and a branch in `InverseProblem.apply`. Write the operator **once**,
against the three-primitive backend shim, so NumPy builds `y` and Torch/JAX evaluate the
objective from the same code.

If it needs randomness (a mask, a noise realisation, a geometry), derive it from a
`semantic_seed` over `(problem, canonical params, image id)` — never from call order. That
is what makes one measurement provably shared by every model and method.

Add validation in `_validate_problem_params`, and confirm the new operator passes
`operator_parity`, `measurement`, and `gradient` in the structural checks.

---

## What not to break

These invariants are what make the numbers comparable. Each is asserted by a check in
`src/checks.py`:

1. **One measurement per problem instance**, shared by every model and method.
2. **One epsilon per `(model, image, replicate)`** — never dependent on method, solver,
   steps, or any hyperparameter.
3. **One initial state** per `(model, problem, t0)`, built by `build_initial_state` and
   bit-identical across strategies.
4. **One conditioning object** per image, identical across strategies.
5. **No graph-breaking operation** inside a differentiable objective.

If you add a strategy that genuinely cannot honour one of these — a method that needs its
own noise schedule, say — record the deviation in the result metadata rather than quietly
letting the comparison drift.



