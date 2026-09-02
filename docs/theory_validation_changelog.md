# RHSO theory-validation suite — changelog, how to run, and the new result fields

Everything below was added for the RHSO theory-validation experiments. The scientific
behaviour of every configuration that existed before is unchanged: `configs/experiments_imagenet100_final.yaml`
is untouched, its 90 resolved job ids are bitwise identical to what they were (verified),
and no existing method, problem, learning rate or default was modified.

---

## 1. Changelog

### Modified

| file | what changed |
|---|---|
| `src/models/base.py` | `StandardFlowAdapter` gains `clean_prediction(state, s, cond)` (generic implementation: `x + (1-s)·v`, honestly labelled `velocity_extrapolation`) and `terminal_prediction(...)`. `MeanFlowAdapter` gains `terminal_prediction(...) = transition(x, s, 1.0, cond)`. Both carry a `terminal_prediction_kind` label so the two families' terminal objects are never described as one. |
| `src/models/jit.py` | Refactored so ONE `_guided_clean` implements the guided clean prediction (CFG, per-sample interval gating, dtype policy, native time). `velocity()` now derives `v = (x̂₁ − x)/max(1−t, t_eps)` from it, and the new `clean_prediction()` returns it directly. New sanity check `direct_clean_prediction`. No numerical change to `velocity`. |
| `src/problems.py` | New `make_phi_per_sample` (per-image fidelity, `sum_b == make_phi`, never differentiated). `InverseProblem` gains `padded_rows`, and `subset(indices, padded_rows=0)` records it. |
| `src/rhso.py` | Terminal-mode resolution (`auto`/`direct`/`suffix`), the shared `make_terminal_planner` factory both loops build their objective from, diagnostic wiring in both loops, `_prepare`/`_finalise` helpers, and `rhso_planning_evaluations` / `rhso_cost_estimate` extended for direct standard-flow planning. Execution semantics unchanged. |
| `src/sdedit.py` | `ReconstructionStats` gains `diagnostic_model_evals`, `diagnostic_network_forwards`, `diagnostic_jvps`, `diagnostic_vjps`, `diagnostic_seconds`, `stage_records`, `consistency_kind`, `terminal_planner_kind`, `diagnostic_settings`, all merged across chunks. |
| `src/config.py` | Seven new RHSO-only fields declared, defaulted, validated and swept; `JobSpec` gains the resolved mode, the planner and consistency labels and the diagnostic settings; job identity extended **only** for jobs that use a new feature; `leaf_dir`, `label`, the dry-run report and the reminders updated; `rhso_terminal_planner()` helper. |
| `src/memory.py` | Adds the cross-framework `gpu_process_*` metric (NVML, fresh sampler per job, framework-correct sync), for **both** Torch and JAX. Process sampling no longer depends on the framework-specific path succeeding. JAX's lifetime high-water mark is never substituted for it. |
| `src/checks.py` | New real-model check `rhso_direct_terminal_planning` (one prediction per objective, no suffix integration, `x+(1-s)v` agreement, diagnostics kept out of the algorithmic counters). |
| `run.py` | New result columns; diagnostic counters and stage summaries in `build_record`; `[image, stage]` arrays plus an `rhso_diagnostics` metadata block in `persist_job`; `padded_rows` passed to `subset`; warm-up key and warm-up reductions extended; semantic resume tolerance for the newly-added spec fields; two new metadata policy strings. |
| `docs/schedule_and_rhso.md` | New section **D** (~200 lines): direct terminal planning, the three diagnostic families, honest Jacobian limits, batch-4 memory/runtime interpretation, the matched-budget design, persistence. |
| `tests/test_rhso.py` | Two checks updated for the refactor (the `mu` resolution moved into the shared `_prepare`; the declared RHSO field set grew). |
| `tests/README.md` | Documents the new test script and states explicitly what JiT-direct coverage is structural. |

### Added

| file | what it is |
|---|---|
| `src/rhso_diagnostics.py` | The whole diagnostic layer: settings resolution, the per-stage recorder, consistency labelling and its written meaning, matrix-free Jacobian probes (JAX + Torch), gain statistics, `stage_arrays` and `stage_summary`. |
| `tests/test_rhso_theory.py` | 33 executable checks covering the 12 requested areas. |
| `configs/experiments_theory_validation_final100.yaml` | The main suite: 70 jobs. |
| `configs/experiments_theory_jacobian_8img.yaml` | 8 images, spectral probes on: 10 jobs. |
| `configs/experiments_theory_resources_batch4.yaml` | 16 images, all diagnostics off: 10 jobs. |
| `configs/experiments_theory_smoke.yaml` | 4 jobs, minutes, for GPU plumbing verification. |
| `docs/theory_validation_changelog.md` | This file. |

### Not done, deliberately

No model retrained, no inverse problem changed, no learning rate retuned, no method deleted,
no new JiT-**suffix** experiment, no gradient clipping, no change to `mu = 0` defaults, no
change to batch reduction or initialisation semantics, and no redesign of the sweep engine.

---

## 2. How to run

Replace `<CONFIG>` with one of the four new files. Start with the smoke config.

```bash
# 1. Dry run: validate, resolve, print the plan. No model is loaded.
python run.py --config <CONFIG> --dry-run

# 2. Prefetch once on a login node (compute nodes have no internet)
python run.py --config <CONFIG> --prefetch --cache-root /path/to/cache

# 3. Local / single GPU
python run.py --config <CONFIG>
python run.py --config configs/experiments_theory_smoke.yaml --check   # + real-model checks

# 4. SLURM, one GPU for the whole config
MPCFLOW_CONFIG=<CONFIG> bash submit.sh
MPCFLOW_CONFIG=<CONFIG> bash submit.sh --time 12:00:00 --mem 96G       # override anything

# 5. SLURM job array (the 70-job suite wants this)
MPCFLOW_CONFIG=<CONFIG> bash submit.sh --array 8

# 6. Merge the array's shards into one results.csv plus figures
python run.py --config <CONFIG> --run-id run_<ARRAYJOBID> --aggregate
```

Suggested order and rough shape:

| config | jobs | purpose |
|---|---|---|
| `experiments_theory_smoke.yaml` | 4 | verify plumbing on a GPU first |
| `experiments_theory_validation_final100.yaml` | 70 | the main result; `--array 8` |
| `experiments_theory_jacobian_8img.yaml` | 10 | anisotropy-vs-stage evidence |
| `experiments_theory_resources_batch4.yaml` | 10 | the memory/runtime table |

Point `data.local_folder` at your copy of the frozen ImageNet-100 benchmark first — all four
configs currently carry the same path as the frozen final config.

Analysis inputs afterwards: `results.csv` (job-level summaries), `results_per_image.csv`
(quality per image), and per job `results.npz` + `metadata.json` for the `[image, stage]`
arrays.

```python
import numpy as np, json
z = np.load("outputs/<run>/<exp>/<model>/rhso/<leaf>/results.npz")
theta = z["stage_theta"]              # [image, stage]
gains = z["stage_jacobian_gains"]     # [image, stage, probe]
meta  = json.load(open(".../metadata.json"))["rhso_diagnostics"]
print(meta["consistency_kind"], meta["consistency_meaning"])
```

---

## 3. New result fields

### `results.csv` (job level)

| column | meaning |
|---|---|
| `rhso_terminal_mode` | resolved `direct` or `suffix` (never `auto`) |
| `rhso_terminal_planner` | `learned_finite_interval_map` (pMF) / `direct_clean_endpoint_prediction` (JiT-direct) / `velocity_extrapolation` / `suffix_integration` |
| `rhso_consistency_kind` | `meanflow_semigroup_defect` or `terminal_prediction_inconsistency`; null when the diagnostic is off |
| `rhso_stage_/consistency_/jacobian_diagnostics` | which diagnostics ran |
| `rhso_jacobian_probes/_power_iters/_seed` | the probe budget and seed actually used |
| `diagnostic_model_evaluations`, `diagnostic_network_forwards`, `diagnostic_jvps`, `diagnostic_vjps`, `diagnostic_seconds` | diagnostic cost, **excluded** from `model_evaluations`, `network_forwards`, `backprops_through_model` and `runtime` |
| `stage_count`, `stage_delta_first/last`, `stage_delta_per_step_first/last`, `stage_theta_first/last`, `stage_v_pre_first`, `stage_v_post_last` | means over images of the per-stage efficiency numbers |
| `endpoint_shift_native_relative_mean`, `endpoint_shift_pixel_rmse_mean`, `fidelity_shift_execution_mean`, `next_stage_recovery_mean` | means over the defined (non-final) stages |
| `jacobian_sigma_max_first/last`, `jacobian_log_anisotropy_first/last` | first- vs last-stage anisotropy, the headline of claim 3 |
| `gpu_process_baseline_gib`, `gpu_process_peak_gib`, `gpu_process_incremental_peak_gib`, `gpu_process_memory_source` | the JiT-vs-pMF comparable memory metric; a **job peak at batch 4**, never per-image |

### `results.npz` (per job, `[image, stage]` unless stated)

`stage_s_from`, `stage_s_to`, `stage_v_pre`, `stage_v_post`, `stage_delta`,
`stage_delta_per_step`, `stage_theta`, `stage_anchor_penalty_post`, `stage_stage_seconds`,
`stage_endpoint_shift_l2`, `stage_endpoint_shift_native_rmse`,
`stage_endpoint_shift_native_relative`, `stage_endpoint_shift_pixel_rmse`,
`stage_fidelity_shift_execution`, `stage_next_stage_recovery`, `stage_next_stage_theta`,
`stage_jacobian_sigma_max`, `stage_jacobian_gain_{min,max,mean,std,p05,p50,p95}`,
`stage_jacobian_empirical_gain_ratio`, `stage_jacobian_empirical_log_anisotropy`,
`stage_jacobian_gains` (`[image, stage, probe]`), `stage_image_ids` (`[image]`),
`stage_index` (`[stage]`).

Final-stage `endpoint_shift_*`, `fidelity_shift_execution`, `next_stage_recovery` and
`next_stage_theta` are **NaN** — there is no subsequent replanning to observe.

### The evidence chain, in these fields

```
stage_jacobian_empirical_log_anisotropy[i, k]  falls with k
    →  stage_delta_per_step[i, k]  rises with k                      (claims 2 + 3)

stage_v_post[i, k]  →  stage_endpoint_shift_*[i, k]
    →  stage_fidelity_shift_execution[i, k]  →  stage_next_stage_recovery[i, k]   (claim 4)

group results.csv by (problem, model) over num_rhso_steps ∈ {1,2,4,8} at N·M = 160  (claim 1)
group by t0 ∈ {1.0, 0.8, 0.6, 0.4} at N=4, M=40                                     (claim 5)
```

---

## 4. Test results

```
tests/test_beta_schedule.py     37/37
tests/test_flow_trajectory.py   12/12
tests/test_meanflow_pnp_dflow.py 14/14
tests/test_rhso.py              47/47
tests/test_rhso_theory.py       33/33
                        total  143/143
```

Also verified outside the scripts: all four new configs validate, resolve and dry-run
through `run.py`; the main suite resolves to exactly 70 jobs with `(N,M) ∈ {(1,160), (2,80),
(4,40), (8,20)}` and no Cartesian products; the frozen benchmark's 90 job ids are unchanged;
old on-disk artefacts remain resumable while a job asking for direct planning or diagnostics
correctly re-runs; and `persist_job` writes the `[image, stage]` arrays and the metadata
block and reads them back without `allow_pickle`.

## 5. Limitations and caveats

1. **PyTorch could not be installed in the environment these changes were developed in**, so
   `rhso.flow_rhso` and the real `JiTAdapter` were **never executed**. JiT-direct is covered
   structurally (the shared planner factory executed against a JiT-shaped toy adapter; a
   bytecode check that `velocity` and `clean_prediction` both delegate to one
   `_guided_clean`) and by the new `rhso_direct_terminal_planning` check, **which runs only
   on a GPU with the real checkpoint**. Run
   `python run.py --config configs/experiments_theory_smoke.yaml --check` before the large
   jobs. This is the single biggest untested area.
2. **No GPU, no checkpoints, no dataset were available**, so no reconstruction of any real
   image was produced, no NVML path was exercised, and no runtime or memory number here is
   real. The memory tests use an injected fake sampler.
3. **No expensive sweep was launched.** Only dry-runs, unit tests and toy-model runs.
4. The `torch.func.jvp`-free double-backward trick used for Torch Jacobian probes keeps a
   `create_graph=True` graph alive for a stage's probes; at 256×256×3 with `batch_size: 4`
   this is the most memory-hungry thing in the repository. If the 8-image Jacobian config
   OOMs, lower `rhso_jacobian_probes`/`_power_iters` or run it at `batch_size: 2` — the
   probes are per-image and unaffected by the batch.
5. The theory configs run RHSO **only**, so the planner emits one "no paired SDEdit baseline"
   warning per experiment block (50 of the 54 warnings on the main config). That is expected:
   these experiments compare RHSO against itself across `N` and `t0`, not against SDEdit.
6. `empirical_gain_ratio` is a lower bound on the condition number, not the condition number.
   Do not report it as one, and do not report `sigma_max_estimate` as an exact `σ_max`.
7. Job identity is preserved only for configurations that use none of the new features. Any
   job that requests direct planning (on a standard flow) or any diagnostic gets a new id
   and a new output directory — correctly, since it is a different experiment.
8. `results.csv` gained ~30 columns. Anything parsing it positionally rather than by header
   will need updating.

---

# Addendum — the six follow-up theory experiments

Scope: **two small opt-in capabilities, six configurations, one test script.** No
reconstruction algorithm was touched. RHSO, JiT, pMF, D-Flow, the solvers, the optimisers,
the inverse problems, the logging, the Slurm infrastructure and the aggregation behave
exactly as before for every configuration that does not ask for the new features.

## What needed code, and what did not

| Experiment | Needed code? | Why |
| --- | --- | --- |
| 1 — JiT one-shot multi-step suffix | No | The existing D-Flow path already optimises one state, differentiates through the full multi-step trajectory, uses `canonical_time_grid`, and honours `solver: heun` with JiT's final-Euler policy. `steps: 4, num_opt_steps: 160` is the experiment. |
| 2A / 2B — larger `N` | No | `num_rhso_steps` and `num_opt_steps` are already sweepable; matched pairs get one named block each. |
| 3 — noise sweep | **Yes** | The measurement seed included `sigma`, so changing `σ` re-drew `ε`. Added `measurement_noise_group`. |
| 4 — `mu` ablation | No | The state anchor is already implemented, detached, per-stage, reset to the executed state, sweepable, and `mu = 0` is the vanilla objective bitwise. |
| 5 — `beta` schedule | No | `beta` is already a shared, sweepable field and the real stage times are already recorded. |
| 6 — endpoint authority | **Yes** | The existing probes are random-direction only. Added `rhso_gradient_authority_diagnostics`. |

Both new fields are documented in `docs/schedule_and_rhso.md` §E.

## Discrepancies found while verifying the prompt's assumptions

1. **`random_inpaint` masks are seeded with `sigma` in the key.** A σ sweep on that task
   re-draws the mask, so the pairing guarantee `(y − Ax)/σ` constant does **not** hold
   there. Existing mask seeding was deliberately left untouched (changing it would alter
   validated results); the validator now warns whenever a pairing group meets
   `random_inpaint`, and §E.1 states the limitation. Experiment 3 is denoising-only, where
   the operator is the identity, so the required experiment is unaffected.
2. **A Jacobian-only configuration was already a silent no-op**, because the probe sits
   behind `begin_stage`'s `settings.stage` guard. Rather than restructure that guard — which
   would change behaviour for such a configuration — the new authority flag follows the
   precedent the consistency diagnostic already sets and **implies** the stage bookkeeping.
   Nothing about existing configurations changes; all of them already set
   `rhso_stage_diagnostics: true`.
3. **`SEED_RECIPES` was deliberately not extended.** It is copied verbatim into every
   `JobSpec` and compared field-by-field by `run._resolved_spec_matches`, so adding a key
   would have broken `--resume` against artefacts already on disk. The paired recipe lives
   in `utils.PAIRED_MEASUREMENT_SEED_RECIPE` and is written into the problem metadata of the
   runs that use it.

## Additional caveats for this addendum

9. `run.NEW_SPEC_FIELDS` and `_describes_legacy_behaviour` were extended so a stored job
   predating the two new spec fields is still resumable **only** when the new job would run
   the same computation — `measurement_noise_group is None` and the authority flag off. A
   job that now asks for either correctly re-runs.
10. `results.csv` gained 7 columns (the two flags plus five authority summaries). Anything
    parsing it positionally rather than by header needs updating; parsing by header is
    unaffected.
11. `configs/experiments_theory_N_fixedM_final100.yaml` is the most expensive file in the
    suite (`B = 40N` up to 640 iterations per image at `N = 16`, ~7.75× one matched-budget
    task-model column). It is at the full 100 images deliberately. Submit it as a job array
    rather than reducing `num_images`, which would break comparability.
12. **PyTorch is still not installable in this environment**, so `rhso.flow_rhso` and
    `_authority_torch` were not executed. The authority probe's JAX twin is executed against
    an analytic linear map; the Torch path is the same three-step recipe (forward → gradient
    of the fidelity at the endpoint → VJP with that gradient as cotangent). Verify it on a
    GPU with `python run.py --config configs/experiments_theory_gradient_authority.yaml
    --check` before launching the large jobs.
13. The `mu` values in Experiment 4 are a documented order-of-magnitude argument, not
    privileged values. The reasoning is written out in that config's header.
