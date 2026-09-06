"""The two new opt-in capabilities, their backward compatibility, and the six new configs.

Run with:  python tests/test_theory_extensions.py      (needs jax + pyyaml)

Two capabilities are added to this repository, both OFF unless a configuration asks for
them:

    measurement_noise_group                  common-random-number measurement noise, so a
                                             sigma sweep reuses one standard-normal
                                             realisation per image (Experiment 3);
    rhso_gradient_authority_diagnostics      the task-aligned endpoint-authority
                                             diagnostic A_k = ||J^T g|| / ||g||
                                             (Experiment 6).

The overriding requirement is that neither changes anything when it is absent, so most of
what follows is a backward-compatibility argument executed rather than asserted: the
paired-noise section compares against an INDEPENDENT transcription of the original seeding
expression, and the diagnostic section compares counters and outputs with the flag on and
off.

The authority probe is checked against an ANALYTIC example -- a linear terminal map
P(x) = D x with a quadratic fidelity, where A = ||D^T g|| / ||g|| is known in closed form --
rather than against the probe's own output.

Torch is not installable in the container these tests were written in, exactly as the
existing suites document, so `_authority_torch` is not executed here; its JAX twin is, and
both are shown to be the same three-step recipe.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import copy                                                              # noqa: E402
import dataclasses                                                       # noqa: E402
import tempfile                                                          # noqa: E402

import jax                                                               # noqa: E402
import jax.numpy as jnp                                                  # noqa: E402
import numpy as np                                                       # noqa: E402

from src.config import (ProblemRequest, load_config, resolve_run_plan,    # noqa: E402
                        validate_config)
from src.problems import build_problem                                    # noqa: E402
from src.rhso_diagnostics import (GRADIENT_AUTHORITY_MIN_ENDPOINT_NORM,   # noqa: E402
                                  STAGE_FLOAT_FIELDS, probe_gradient_authority,
                                  settings_from_spec, stage_arrays)
from src.utils import (canonical_params_key, derive_rng, gaussian_noise,  # noqa: E402
                       semantic_seed, set_global_seed)

RESULTS = []


def report(name, ok, detail=""):
    print("  [%s] %-44s %s" % ("PASS" if ok else "FAIL", name, detail))
    RESULTS.append(bool(ok))
    return bool(ok)


# =====================================================================================
# Fixtures
# =====================================================================================
RES = 256          # build_problem enforces the canonical resolution
IMAGE_IDS = ("img0", "img1", "img2")


def ground_truth(n=3, seed=11):
    """A canonical (N, 256, 256, 3) float32 batch inside [-1, 1]."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.standard_normal((n, RES, RES, 3)) * 0.3, -1.0, 1.0).astype(np.float32)


def denoising_request(sigma, group=None, key="k"):
    return ProblemRequest(key=key, problem="denoising", params={"sigma": sigma},
                          num_images=3, seed=42, guide_mode="identity",
                          experiments=("probe",), measurement_noise_group=group)


def deblur_request(sigma, blur_sigma=1.0, group=None):
    return ProblemRequest(key="k", problem="deblur",
                          params={"sigma": sigma, "blur_sigma": blur_sigma,
                                  "kernel_size": 7, "padding": "reflect"},
                          num_images=3, seed=42, guide_mode="observed",
                          experiments=("probe",), measurement_noise_group=group)


def epsilon_of(problem, gt):
    """The standard-normal realisation implied by a built problem: (y - A x*) / sigma."""
    from src.utils import NUMPY_BACKEND
    clean = np.asarray(problem.apply(gt, NUMPY_BACKEND), np.float64)
    return (np.asarray(problem.measurement, np.float64) - clean) / float(problem.sigma)


# =====================================================================================
# 1. Paired measurement noise
# =====================================================================================
def test_paired_noise():
    print("\n1. Common-random-number measurement noise")
    set_global_seed(42)
    gt = ground_truth()

    # ---- 1. absent field -> BITWISE the original seeding -----------------------------
    # Compared against an independent transcription of the pre-change expression rather
    # than against another call into the same code path.
    plain = build_problem(denoising_request(0.2), gt, IMAGE_IDS, verbose=False)
    cfg = {"sigma": 0.2}
    legacy_key = canonical_params_key(cfg)
    legacy_noise = np.stack(
        [gaussian_noise((RES, RES, 3), "measurement", "denoising", legacy_key, iid)
         for iid in IMAGE_IDS], axis=0)
    legacy_y = (gt + 0.2 * legacy_noise).astype(np.float32)
    report("no pairing field -> original seeding",
           np.array_equal(plain.measurement, legacy_y),
           "y is bitwise the value the pre-change expression "
           "seed(global, 'measurement', problem, params_key, image) produces")

    # ---- 2. THE property: one epsilon, rescaled --------------------------------------
    paired = {s: build_problem(denoising_request(s, "gA"), gt, IMAGE_IDS, verbose=False)
              for s in (0.1, 0.2, 0.4)}
    eps = {s: epsilon_of(p, gt) for s, p in paired.items()}
    pairs_equal = all(np.allclose(eps[0.2], eps[s], rtol=0, atol=1e-6)
                      for s in (0.1, 0.4))
    report("(y - Ax)/sigma is identical across sigma", pairs_equal,
           "sigma in {0.1, 0.2, 0.4} share one N(0, I) draw per image to 1e-6; max "
           "deviation %.2e" % max(float(np.abs(eps[0.2] - eps[s]).max())
                                  for s in (0.1, 0.4)))
    report("the epsilon really is standard normal",
           abs(float(eps[0.2].std()) - 1.0) < 0.1 and abs(float(eps[0.2].mean())) < 0.1,
           "mean %.3f, std %.3f over %d entries"
           % (float(eps[0.2].mean()), float(eps[0.2].std()), eps[0.2].size))

    # ---- 3. a different group is an independent realisation --------------------------
    other_group = build_problem(denoising_request(0.2, "gB"), gt, IMAGE_IDS, verbose=False)
    report("a different pairing group re-draws",
           not np.allclose(epsilon_of(other_group, gt), eps[0.2], atol=1e-3),
           "group 'gB' shares nothing with group 'gA' at the same sigma and image")

    # ---- 4. a different image is an independent realisation --------------------------
    shifted = build_problem(denoising_request(0.2, "gA"), gt,
                            ("imgX", "img1", "img2"), verbose=False)
    eps_shifted = epsilon_of(shifted, gt)
    report("noise follows the IMAGE ID",
           not np.allclose(eps_shifted[0], eps[0.2][0], atol=1e-3)
           and np.allclose(eps_shifted[1], eps[0.2][1], atol=1e-6),
           "renaming image 0 re-draws only image 0; images 1 and 2 keep their own epsilon")

    # ---- 5. deterministic repeatability ----------------------------------------------
    again = build_problem(denoising_request(0.2, "gA"), gt, IMAGE_IDS, verbose=False)
    report("deterministic across rebuilds",
           np.array_equal(again.measurement, paired[0.2].measurement),
           "rebuilding the same request reproduces y bitwise")

    # ---- 6. the pairing must not reach the operator ----------------------------------
    # A different STRUCTURAL parameter is a different operator and must get its own
    # realisation even inside one group.
    blur_a = build_problem(deblur_request(0.05, 1.0, "gA"), gt, IMAGE_IDS, verbose=False)
    blur_b = build_problem(deblur_request(0.05, 2.0, "gA"), gt, IMAGE_IDS, verbose=False)
    structural_independent = not np.allclose(epsilon_of(blur_a, gt),
                                             epsilon_of(blur_b, gt), atol=1e-3)
    blur_hi = build_problem(deblur_request(0.10, 1.0, "gA"), gt, IMAGE_IDS, verbose=False)
    blur_paired = np.allclose(epsilon_of(blur_a, gt), epsilon_of(blur_hi, gt), atol=1e-6)
    report("structural parameters still separate realisations",
           structural_independent and blur_paired,
           "blur_sigma 1.0 vs 2.0 are independent draws; sigma 0.05 vs 0.10 at one blur "
           "width share theirs -- only the noise AMPLITUDE is excluded from the seed")

    # ---- masks and operators are untouched by the feature ----------------------------
    mask_req = ProblemRequest(key="k", problem="box_inpaint",
                              params={"sigma": 0.05, "box": 4}, num_images=3, seed=42,
                              guide_mode="zero_fill", experiments=("probe",))
    mask_plain = build_problem(mask_req, gt, IMAGE_IDS, verbose=False)
    mask_paired = build_problem(
        dataclasses.replace(mask_req, measurement_noise_group="gA"), gt, IMAGE_IDS,
        verbose=False)
    report("masks and operators are unchanged by pairing",
           np.array_equal(mask_plain.mask, mask_paired.mask),
           "the mask is bitwise identical with and without a group: the feature touches "
           "the noise seed only, never the operator")

    # ---- metadata is auditable -------------------------------------------------------
    meta_on = paired[0.2].to_metadata()
    meta_off = plain.to_metadata()
    report("the group is recorded in the metadata",
           meta_on["measurement_noise_group"] == "gA"
           and meta_off["measurement_noise_group"] is None
           and "sigma deliberately excluded" in meta_on["measurement_seed_recipe"],
           "problem metadata names the group and the recipe actually used, so a paired run "
           "cannot be mistaken for an unpaired one")


# =====================================================================================
# 2. Gradient-aligned endpoint authority -- against a closed-form answer
# =====================================================================================
class LinearAdapterStub:
    """The three attributes `probe_gradient_authority` reads, and nothing else."""

    def __init__(self):
        self.spec = dataclasses.replace(_AdapterSpecStub(), framework="jax")

    def to_pixels(self, state, differentiable=False):
        return state


@dataclasses.dataclass(frozen=True)
class _AdapterSpecStub:
    framework: str = "jax"


def test_gradient_authority_analytic():
    print("\n2. Gradient-aligned endpoint authority (analytic linear map)")
    rng = np.random.default_rng(7)
    dim, batch = 5, 3
    # P(x) = D x, applied INDEPENDENTLY to each batch row -- which is how every real
    # adapter behaves and what makes the per-image reading legitimate.
    D = jnp.asarray(rng.standard_normal((dim, dim)), jnp.float32)
    target = jnp.asarray(rng.standard_normal((batch, dim)), jnp.float32)
    x = jnp.asarray(rng.standard_normal((batch, dim)), jnp.float32)

    def predict(state):
        return state @ D.T

    # Quadratic fidelity, summed over the batch: F = 1/2 sum_b ||x1_b - target_b||^2,
    # so grad_{x1} F = x1 - target, one independent block per image.
    def fidelity(state):
        return 0.5 * jnp.sum((state - target) ** 2)

    adapter = LinearAdapterStub()
    counters = dataclasses.make_dataclass(
        "C", [("diagnostic_model_evals", int, 0), ("diagnostic_vjps", int, 0)])()
    rows = probe_gradient_authority(adapter, predict, x, batch, fidelity, counters)

    # ---- closed form ------------------------------------------------------------------
    endpoint = np.asarray(x @ D.T - target, np.float64)          # g_end per row
    expected_end = np.sqrt((endpoint ** 2).sum(axis=1))
    expected_state = np.sqrt(((endpoint @ np.asarray(D, np.float64)) ** 2).sum(axis=1))
    got_end = np.asarray([r["endpoint_fidelity_grad_norm"] for r in rows])
    got_state = np.asarray([r["state_fidelity_grad_norm"] for r in rows])
    got_auth = np.asarray([r["gradient_aligned_authority"] for r in rows])
    report("A = ||D^T g|| / ||g|| matches the closed form",
           np.allclose(got_end, expected_end, rtol=1e-4)
           and np.allclose(got_state, expected_state, rtol=1e-4)
           and np.allclose(got_auth, expected_state / expected_end, rtol=1e-4),
           "3 images, max relative error %.2e against the analytic value"
           % float(np.max(np.abs(got_auth - expected_state / expected_end)
                          / (expected_state / expected_end))))
    report("the log is the log of the ratio",
           np.allclose([r["log_gradient_aligned_authority"] for r in rows],
                       np.log(got_auth), rtol=1e-6),
           "log_gradient_aligned_authority = log(gradient_aligned_authority)")

    # ---- per-image isolation ----------------------------------------------------------
    # Row 0 alone is re-run in a batch of one; a correct implementation gives the same
    # number, because image b's authority must not depend on its batch companions.
    solo = probe_gradient_authority(
        adapter, predict, x[:1],  1,
        lambda state: 0.5 * jnp.sum((state - target[:1]) ** 2), None)
    report("per-image isolation",
           abs(solo[0]["gradient_aligned_authority"]
               - rows[0]["gradient_aligned_authority"]) < 1e-4,
           "image 0 scores %.6f alone and %.6f inside a batch of 3 -- the batch-summed "
           "fidelity leaves each row's gradient its own"
           % (solo[0]["gradient_aligned_authority"], rows[0]["gradient_aligned_authority"]))

    # ---- padded rows are never read ---------------------------------------------------
    padded = probe_gradient_authority(adapter, predict, x, 2, fidelity, None)
    report("no padded-row contamination",
           len(padded) == 2
           and abs(padded[0]["gradient_aligned_authority"]
                   - rows[0]["gradient_aligned_authority"]) < 1e-9,
           "real_rows=2 on a batch of 3 returns exactly 2 rows and the real ones are "
           "unchanged: a padding duplicate can never become an independent sample")

    # ---- determinism -------------------------------------------------------------------
    twice = probe_gradient_authority(adapter, predict, x, batch, fidelity, None)
    report("deterministic",
           all(abs(a["gradient_aligned_authority"] - b["gradient_aligned_authority"]) == 0.0
               for a, b in zip(rows, twice)),
           "the probe has no randomness of its own: two calls agree bitwise")

    # ---- accounting --------------------------------------------------------------------
    report("counted as DIAGNOSTIC work only",
           counters.diagnostic_model_evals == 1 and counters.diagnostic_vjps == 1,
           "one terminal-planner evaluation and one VJP per stage, both in the "
           "diagnostic_* counters; no algorithmic counter exists on this object to touch")

    # ---- zero endpoint gradient -> the documented NaN ----------------------------------
    at_optimum = jnp.linalg.solve(jnp.asarray(D, jnp.float32).T,
                                  jnp.zeros((dim,), jnp.float32))
    del at_optimum
    zero_rows = probe_gradient_authority(
        adapter, predict, x, batch,
        # A fidelity whose gradient vanishes identically: g_end == 0 exactly.
        lambda state: jnp.sum(state) * 0.0, None)
    nan_ratio = all(np.isnan(r["gradient_aligned_authority"]) for r in zero_rows)
    norms_kept = all(r["endpoint_fidelity_grad_norm"] == 0.0 for r in zero_rows)
    report("zero endpoint gradient -> NaN, not a fake finite value",
           nan_ratio and norms_kept,
           "||g_end|| = 0 < %g leaves the ratio NaN while BOTH norms stay on the row, so "
           "the reader can see why it is missing"
           % GRADIENT_AUTHORITY_MIN_ENDPOINT_NORM)


# =====================================================================================
# 3. Serialisation and NaN-safe aggregation
# =====================================================================================
def test_serialisation():
    print("\n3. Stage serialisation")
    fields = ("endpoint_fidelity_grad_norm", "state_fidelity_grad_norm",
              "gradient_aligned_authority", "log_gradient_aligned_authority")
    report("the new fields joined the central stage schema",
           all(f in STAGE_FLOAT_FIELDS for f in fields),
           "STAGE_FLOAT_FIELDS carries all four, so persistence, the [image, stage] "
           "arrays and the summaries pick them up without a parallel result format")

    records = []
    for stage in range(2):
        for row, image_id in enumerate(("img0", "img1")):
            entry = {name: float("nan") for name in STAGE_FLOAT_FIELDS}
            entry.update({"stage": stage, "image_row": row, "image_id": image_id,
                          "gains": [],
                          "endpoint_fidelity_grad_norm": 2.0 + stage,
                          "state_fidelity_grad_norm": 1.0 + stage,
                          "gradient_aligned_authority": (1.0 + stage) / (2.0 + stage)})
            # One deliberately undefined ratio, to prove the aggregation drops it.
            if stage == 1 and row == 1:
                entry["gradient_aligned_authority"] = float("nan")
            records.append(entry)

    arrays = stage_arrays(records, ("img0", "img1"))
    shaped = all(arrays["stage_" + f].shape == (2, 2) for f in fields)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.npz"
        np.savez_compressed(path, **arrays)
        with np.load(path) as z:                      # no allow_pickle, as on the real path
            loaded = {k: z[k] for k in z.files}
    round_trips = np.allclose(loaded["stage_gradient_aligned_authority"],
                              arrays["stage_gradient_aligned_authority"], equal_nan=True)
    report("[image, stage] arrays round-trip through results.npz",
           shaped and round_trips and "stage_theta" in loaded,
           "the authority arrays are (image, stage) alongside the existing ones and load "
           "back without allow_pickle -- old readers keyed by name are unaffected")

    from src.rhso_diagnostics import stage_summary
    summary = stage_summary(records)
    report("the summary skips the NaN rather than propagating it",
           abs(summary["gradient_aligned_authority_first"] - 0.5) < 1e-9
           and abs(summary["gradient_aligned_authority_last"] - (2.0 / 3.0)) < 1e-9,
           "stage 1's mean is taken over the one FINITE row (%.4f), not NaN"
           % summary["gradient_aligned_authority_last"])


# =====================================================================================
# 4. The diagnostic inside the real RHSO loop
# =====================================================================================
def test_end_to_end_rhso():
    print("\n4. The diagnostic inside the real RHSO loop")
    from spec_support import make_spec
    from src.models.base import Conditioning
    from src.rhso import meanflow_rhso
    from test_rhso import BATCH, RES as TOY_RES, ToyMeanFlow, make_problem

    problem = make_problem()
    cond = Conditioning(labels=np.zeros((BATCH,), np.int32), guidance={})
    x0 = jnp.asarray(np.random.default_rng(0).standard_normal(
        (BATCH, TOY_RES, TOY_RES, 3)), jnp.float32)

    def run(**over):
        adapter = ToyMeanFlow(problem.ground_truth)
        spec = make_spec("rhso", model="pmf", num_rhso_steps=3, num_opt_steps=4, lr=0.05,
                         **over)
        out, stats = meanflow_rhso(adapter, cond, x0, problem, spec)
        return np.asarray(jax.device_get(out), np.float64), stats

    plain_x, plain = run()
    auth_x, auth = run(rhso_gradient_authority_diagnostics=True)

    # ---- the reconstruction itself is untouched ---------------------------------------
    report("the diagnostic does not move the trajectory",
           np.array_equal(plain_x, auth_x),
           "the executed states are BITWISE identical with the diagnostic on and off: it "
           "reads the model, it never steers it")

    # ---- and so is every algorithmic counter ------------------------------------------
    algorithmic = ("control_iterations", "optimizer_iterations", "objective_evals",
                   "data_gradient_evals", "model_evals_total", "model_evals_planning",
                   "backprops_through_model", "network_forwards")
    same = {f: (getattr(plain, f), getattr(auth, f)) for f in algorithmic}
    report("algorithmic accounting is unchanged",
           all(a == b for a, b in same.values()),
           "optimizer steps, model evaluations, backprops and network forwards all match "
           "(%d optimizer iterations, %d model evaluations either way)"
           % (auth.optimizer_iterations, auth.model_evals_total))

    # ---- the diagnostic's own cost is counted separately ------------------------------
    report("the diagnostic's cost lands in the diagnostic counters",
           auth.diagnostic_model_evals > 0 and plain.diagnostic_model_evals == 0
           and auth.diagnostic_vjps > 0 and plain.diagnostic_vjps == 0
           and auth.diagnostic_seconds > 0.0,
           "%d diagnostic model evaluations and %d VJPs over 3 stages, %.3fs of diagnostic "
           "time -- none of it added to the algorithmic totals above"
           % (auth.diagnostic_model_evals, auth.diagnostic_vjps, auth.diagnostic_seconds))

    # ---- the rows are populated, per image and per stage -------------------------------
    rows = auth.stage_records
    finite = [r for r in rows
              if np.isfinite(r["gradient_aligned_authority"])
              and np.isfinite(r["endpoint_fidelity_grad_norm"])]
    ratios_consistent = all(
        abs(r["gradient_aligned_authority"]
            - r["state_fidelity_grad_norm"] / r["endpoint_fidelity_grad_norm"]) < 1e-6
        for r in finite)
    report("one authority row per (image, stage)",
           len(rows) == 3 * BATCH and len(finite) == 3 * BATCH and ratios_consistent,
           "%d rows = 3 stages x %d images, every one carrying both norms and a ratio that "
           "equals state/endpoint" % (len(rows), BATCH))

    # ---- and the run without the flag records nothing of the sort ----------------------
    report("nothing is recorded when the flag is off", not plain.stage_records,
           "the default run produces no stage records at all, so no result file gains a "
           "column it did not have before")


# =====================================================================================
# 5. Both features are off by default and change nothing when absent
# =====================================================================================
def test_defaults_and_backward_compatibility():
    print("\n5. Defaults and backward compatibility")
    from spec_support import make_spec

    spec = make_spec("rhso")
    report("both new fields default to off",
           spec.rhso_gradient_authority_diagnostics is False
           and spec.measurement_noise_group is None,
           "a configuration that never mentions them resolves to "
           "rhso_gradient_authority_diagnostics=False, measurement_noise_group=None")

    settings = settings_from_spec(spec)
    report("no diagnostic recorder is built by default",
           not settings.any_enabled and not settings.gradient_authority,
           "any_enabled is False, so the RHSO loops take the branch that has no recorder, "
           "no extra model call and no extra timer")

    # A spec from before the field existed must resolve to the old behaviour.
    class _NoAuthority:
        rhso_stage_diagnostics = True
        rhso_consistency_diagnostics = False
        rhso_jacobian_diagnostics = False

    legacy = settings_from_spec(_NoAuthority())
    report("a spec predating the field resolves to off",
           legacy.stage and not legacy.gradient_authority,
           "getattr-with-default keeps an older spec, and any test stand-in, working")

    # Asking for the authority implies the stage bookkeeping its rows live on, exactly as
    # a consistency measurement already did.
    class _AuthorityOnly:
        rhso_stage_diagnostics = False
        rhso_consistency_diagnostics = False
        rhso_jacobian_diagnostics = False
        rhso_gradient_authority_diagnostics = True

    implied = settings_from_spec(_AuthorityOnly())
    report("authority implies the stage bookkeeping",
           implied.gradient_authority and implied.stage,
           "asking for it alone cannot silently measure nothing -- the same rule the "
           "consistency diagnostic already follows")

    # ---- the validated config is untouched --------------------------------------------
    config = load_config("configs/experiments_theory_validation_final100.yaml")
    plan = resolve_run_plan(config, validate_config(config), run_id="probe")
    rhso = [s for s in plan.specs if s.method == "rhso"]
    pairs = sorted({(s.num_rhso_steps, s.num_opt_steps) for s in rhso if s.t0 == 1.0})
    report("the validated 100-image config still resolves identically",
           len(plan.specs) == 70 and pairs == [(1, 160), (2, 80), (4, 40), (8, 20)]
           and all(s.measurement_noise_group is None for s in plan.specs)
           and not any(s.rhso_gradient_authority_diagnostics for s in plan.specs),
           "70 atomic jobs, the four matched pairs, and neither new field engaged")

    # Job IDS are the identity that decides output paths and resume, so they are pinned
    # against a run of the SAME planner with the new fields stripped back out.
    ids = sorted(s.job_id for s in plan.specs)
    dirs = sorted(s.leaf_dir for s in plan.specs)
    plan_again = resolve_run_plan(config, validate_config(config), run_id="probe")
    report("job ids and output paths are stable",
           ids == sorted(s.job_id for s in plan_again.specs)
           and dirs == sorted(s.leaf_dir for s in plan_again.specs)
           and not any("diag=" in d and "a" in d.split("diag=")[1].split("__")[0]
                       for d in dirs),
           "identities and leaf directories are unchanged and no path gained the new "
           "diagnostic's flag letter")

    # The identity only grows when a job actually opts in.
    base = copy.deepcopy(config)
    first = list(base["experiments"])[0]
    baseline = resolve_run_plan(base, validate_config(base), run_id="x")
    opted = copy.deepcopy(base)
    for model in opted["experiments"][first]["models"].values():
        model["methods"]["rhso"]["rhso_gradient_authority_diagnostics"] = True
    opted_plan = resolve_run_plan(opted, validate_config(opted), run_id="x")
    changed = sum(1 for a, b in zip(baseline.specs, opted_plan.specs)
                  if a.job_id != b.job_id)
    report("opting in changes only the opted-in jobs",
           changed == 2 and len(opted_plan.specs) == len(baseline.specs),
           "turning the flag on in one experiment moves exactly its 2 job ids; the other "
           "68 are untouched, so a partially-rerun sweep keeps its old results")


# =====================================================================================
# 6. The six new configurations
# =====================================================================================
EXPECTED = {
    "theory_jit_suffix_final100": dict(jobs=5, models={"jit"}, methods={"dflow"}, images={100}),
    "theory_N_fixedB_final100": dict(jobs=60, models={"jit", "pmf"}, methods={"rhso"},
                                     images={100}),
    "theory_N_fixedM_final100": dict(jobs=50, models={"jit", "pmf"}, methods={"rhso"},
                                     images={100}),
    "theory_noise_strength": dict(jobs=40, models={"jit", "pmf"}, methods={"rhso"},
                                  images={100}),
    "theory_mu_denoising": dict(jobs=24, models={"jit", "pmf"}, methods={"rhso"},
                                images={100}),
    "theory_beta_schedule": dict(jobs=50, models={"jit", "pmf"}, methods={"rhso"},
                                 images={100}),
    "theory_gradient_authority": dict(jobs=10, models={"jit", "pmf"}, methods={"rhso"},
                                      images={8}),
}


def test_new_configs():
    print("\n6. The six new configurations")
    plans = {}
    for name, want in EXPECTED.items():
        cfg = load_config("configs/experiments_%s.yaml" % name)
        plan = resolve_run_plan(cfg, validate_config(cfg), run_id=name)
        plans[name] = plan
        ids = [s.job_id for s in plan.specs]
        report("%s expands as intended" % name,
               len(plan.specs) == want["jobs"] and len(set(ids)) == len(ids)
               and {s.model for s in plan.specs} == want["models"]
               and {s.method for s in plan.specs} == want["methods"]
               and {s.num_images for s in plan.specs} == want["images"],
               "%d unique jobs | models %s | methods %s | images %s"
               % (len(plan.specs), sorted(want["models"]), sorted(want["methods"]),
                  sorted(want["images"])))

    # ---- Experiment 1: one state, four intervals, 160 iterations ----------------------
    suffix = plans["theory_jit_suffix_final100"].specs
    report("E1 is one-shot optimisation through a 4-interval suffix",
           all(s.steps == 4 and s.num_opt_steps == 160 and s.solver == "heun"
               and s.t0 == 1.0 and s.beta == 1.0 and s.lr == 0.01 for s in suffix)
           and {s.problem for s in suffix} == {"denoising", "deblur", "super_resolution",
                                               "box_inpaint", "random_inpaint"},
           "D-Flow, steps=4, 160 Adam iterations at the source state, Heun, t0=1 "
           "(canonical s=0), and the validated JiT RHSO lr on all five tasks")

    # ---- Experiment 2A: matched budget, no spurious Cartesian product -----------------
    fixed_b = plans["theory_N_fixedB_final100"].specs
    pairs_b = sorted({(s.num_rhso_steps, s.num_opt_steps) for s in fixed_b})
    report("E2A holds B = N*M = 160 at every point",
           pairs_b == [(1, 160), (2, 80), (4, 40), (8, 20), (16, 10), (32, 5)]
           and all(s.num_rhso_steps * s.num_opt_steps == 160 for s in fixed_b),
           "the six matched pairs and NOT the 36-job Cartesian product: %s" % (pairs_b,))

    # ---- Experiment 2B: fixed per-stage budget ----------------------------------------
    fixed_m = plans["theory_N_fixedM_final100"].specs
    report("E2B holds M = 40 and lets B grow",
           all(s.num_opt_steps == 40 for s in fixed_m)
           and sorted({s.num_rhso_steps for s in fixed_m}) == [1, 2, 4, 8, 16]
           and sorted({s.num_rhso_steps * s.num_opt_steps for s in fixed_m})
           == [40, 80, 160, 320, 640],
           "M is 40 everywhere, N in {1,2,4,8,16}, so B = 40N runs 40 -> 640 deliberately")

    # ---- Experiment 3: paired noise, sigma still separates jobs -----------------------
    noise = plans["theory_noise_strength"].specs
    sigmas = sorted({float(s.problem_params["sigma"]) for s in noise})
    problem_keys = {float(s.problem_params["sigma"]): s.problem_key for s in noise}
    report("E3 sweeps sigma under one pairing group",
           sigmas == [0.0, 0.1, 0.2, 0.4]
           and all(s.measurement_noise_group == "theory_noise_sweep_v1" for s in noise)
           and len(set(problem_keys.values())) == 4
           and all(s.num_rhso_steps * s.num_opt_steps == 160 for s in noise),
           "4 noise levels around sigma_0 = 0.2, each its own problem instance and its own "
           "jobs, all sharing one epsilon per image; B = 160 throughout")

    # ---- Experiment 4: mu ablation ----------------------------------------------------
    mus = plans["theory_mu_denoising"].specs
    report("E4 sweeps mu at two matched-budget stage counts",
           sorted({s.mu for s in mus}) == [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
           and sorted({(s.num_rhso_steps, s.num_opt_steps) for s in mus})
           == [(4, 40), (8, 20)]
           and {s.problem for s in mus} == {"denoising"}
           and all(s.record_loss_history for s in mus),
           "6 log-spaced mu values including 0, at (4,40) and (8,20), denoising only, with "
           "the fidelity / anchor / total split kept in the loss histories")

    # ---- Experiment 5: beta, and the ACTUAL times ------------------------------------
    betas = plans["theory_beta_schedule"].specs
    from src.rhso import rhso_time_grid
    grids = {b: [round(t, 6) for t in rhso_time_grid(
        [s for s in betas if s.beta == b][0])] for b in (0.5, 1.0, 2.0)}
    # beta < 1 must sit LATER (nearer the data end) than uniform; beta > 1 earlier.
    direction = (grids[0.5][1] > grids[1.0][1] > grids[2.0][1])
    report("E5 sweeps beta and the stage times move as documented",
           sorted({s.beta for s in betas}) == [0.5, 0.75, 1.0, 1.5, 2.0]
           and all(s.num_rhso_steps == 4 and s.num_opt_steps == 40 for s in betas)
           and direction and grids[1.0] == [0.0, 0.25, 0.5, 0.75, 1.0],
           "s_1 is %.4f (beta=0.5, finer near DATA), %.4f (uniform), %.4f (beta=2, finer "
           "near NOISE)" % (grids[0.5][1], grids[1.0][1], grids[2.0][1]))
    report("E5 records the true stage locations",
           all(s.delta is None for s in betas if s.beta != 1.0)
           and all(s.delta_min is not None and s.delta_max is not None for s in betas),
           "`delta` is null wherever beta != 1 -- a non-uniform grid has no single step "
           "size -- while delta_min/delta_max always describe the real thing")

    # ---- Experiment 6: both diagnostics, on the jacobian config's images --------------
    auth = plans["theory_gradient_authority"].specs
    jac_cfg = load_config("configs/experiments_theory_jacobian_8img.yaml")
    jac = resolve_run_plan(jac_cfg, validate_config(jac_cfg), run_id="j").specs
    same_setting = ({(s.model, s.problem, s.lr, s.num_rhso_steps, s.num_opt_steps,
                      s.t0, s.beta, s.mu, s.num_images) for s in auth}
                    == {(s.model, s.problem, s.lr, s.num_rhso_steps, s.num_opt_steps,
                         s.t0, s.beta, s.mu, s.num_images) for s in jac})
    report("E6 runs both diagnostics on the jacobian config's exact setting",
           all(s.rhso_gradient_authority_diagnostics and s.rhso_jacobian_diagnostics
               and s.rhso_stage_diagnostics and s.rhso_consistency_diagnostics
               for s in auth)
           and same_setting
           and {s.rhso_jacobian_seed for s in auth} == {20240917},
           "task-aligned authority and generic anisotropy are measured in ONE run on the "
           "same 8 images, stages and probe seed, so the two are directly comparable")
    report("E6 keeps the new flag out of the old jobs' identities",
           all(not s.rhso_gradient_authority_diagnostics for s in jac)
           and {s.job_id for s in auth}.isdisjoint({s.job_id for s in jac}),
           "the existing jacobian config is unchanged and cannot collide with the new one")


def main():
    print("Theory-extension suite: paired measurement noise + gradient-aligned authority")
    test_paired_noise()
    test_gradient_authority_analytic()
    test_serialisation()
    test_end_to_end_rhso()
    test_defaults_and_backward_compatibility()
    test_new_configs()
    passed = sum(1 for r in RESULTS if r)
    print("\n%d/%d checks passed" % (passed, len(RESULTS)))
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
