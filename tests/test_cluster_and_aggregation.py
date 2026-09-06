"""Regression tests for the bugs found by the first real Rorqual H100 smoke run.

Every test here corresponds to something that actually went wrong on the cluster, so each
one is written to fail loudly if the old behaviour comes back:

  1. `_probe_spec` raising TypeError when a caller overrides one of its defaults;
  2. the direct-terminal-planning check inheriting diagnostics from a theory config and
     then requiring the baseline to have none;
  3. array shards overwriting each other's checks.json / run_metadata.json;
  4. `--aggregate` producing results.csv but never results_per_image.csv;
  5. the stale "differentiates the whole REMAINING suffix" warning on JiT-direct jobs;
  6. Rorqual silently using the default CUDA 12.6.2 compiler;
  7. total DEVICE memory standing in for PROCESS memory.

Runs on CPU with no checkpoint and no GPU:

    python tests/test_cluster_and_aggregation.py
"""

import dataclasses
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PASSED = [0]
FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    if ok:
        PASSED[0] += 1
    else:
        FAILED.append(name)


# =====================================================================================
# 1. BUG 1 -- _probe_spec must let an explicit override win
# =====================================================================================
def test_probe_spec_override() -> None:
    print("\n1. checks._probe_spec: explicit overrides win, no duplicate keyword")
    from tests.spec_support import make_spec

    spec = make_spec("rhso", model="pmf", num_rhso_steps=2, num_opt_steps=2)

    # An exact transcription of the helper as it now stands in src/checks.py: defaults and
    # overrides merged into ONE mapping before dataclasses.replace is called.
    def probe(base, method, n, **overrides):
        fields = {"method": method, "num_images": n, "record_loss_history": False}
        fields.update(overrides)
        return dataclasses.replace(base, **fields)

    # The old form, kept here so the test proves the bug was real rather than assumed.
    def probe_old(base, method, n, **overrides):
        return dataclasses.replace(base, method=method, num_images=n,
                                   record_loss_history=False, **overrides)

    raised = False
    try:
        probe_old(spec, "rhso", 2, record_loss_history=True)
    except TypeError as exc:
        raised = "multiple values" in str(exc)
    check("the old form really did raise TypeError", raised,
          "dataclasses.replace() got multiple values for 'record_loss_history'")

    fixed = probe(spec, "rhso", 2, record_loss_history=True)
    check("record_loss_history=True override wins", fixed.record_loss_history is True,
          "the exact call rhso_receding_horizon and rhso_state_regularization make")

    default = probe(spec, "rhso", 2)
    check("the default still applies when not overridden",
          default.record_loss_history is False)

    # Not special-cased to that one field: EVERY default must be overridable.
    other = probe(spec, "rhso", 2, num_images=7)
    check("num_images is overridable too", other.num_images == 7,
          "the fix is general, not a two-caller patch")

    both = probe(spec, "rhso", 2, record_loss_history=True, mu=0.5, num_opt_steps=3)
    check("mixed overrides and defaults compose",
          both.record_loss_history is True and both.mu == 0.5 and both.num_opt_steps == 3)


# =====================================================================================
# 2. BUG 2 -- the direct-planning check's baseline must disable diagnostics itself
# =====================================================================================
def test_direct_planning_probe_disables_diagnostics() -> None:
    print("\n2. rhso_direct_terminal_planning: the PLAIN probe has diagnostics OFF")
    from src.rhso_diagnostics import settings_from_spec
    from tests.spec_support import make_spec

    # A spec shaped like the smoke config: theory diagnostics ON, exactly as
    # configs/experiments_theory_smoke.yaml resolves them.
    smoke = make_spec("rhso", model="pmf", num_rhso_steps=2, num_opt_steps=2,
                      rhso_stage_diagnostics=True, rhso_consistency_diagnostics=True,
                      rhso_jacobian_diagnostics=True)
    check("the smoke-shaped spec really has diagnostics on",
          settings_from_spec(smoke).any_enabled)

    # The OLD probe only set the terminal mode, so it inherited all three flags and the
    # check's `plain.diagnostic_model_evals == 0` requirement could not hold.
    old_probe = dataclasses.replace(smoke, rhso_terminal_mode="direct")
    check("the old probe really did inherit them",
          settings_from_spec(old_probe).any_enabled,
          "which is why the check failed on a passing run")

    # The NEW probe, transcribed from src/checks.py.
    new_probe = dataclasses.replace(
        smoke, method="rhso", num_images=2, record_loss_history=False,
        rhso_terminal_mode="direct", rhso_stage_diagnostics=False,
        rhso_consistency_diagnostics=False, rhso_jacobian_diagnostics=False)
    settings = settings_from_spec(new_probe)
    check("the baseline probe has ALL diagnostics off", not settings.any_enabled,
          "stage=%s consistency=%s jacobian=%s"
          % (settings.stage, settings.consistency, settings.jacobian))
    check("the baseline keeps terminal_mode=direct",
          new_probe.rhso_terminal_mode == "direct")

    # And the diagnostics-enabled companion is built FROM that baseline deliberately.
    with_diag = dataclasses.replace(new_probe, rhso_stage_diagnostics=True,
                                    rhso_consistency_diagnostics=True)
    enabled = settings_from_spec(with_diag)
    check("the companion probe turns them back on",
          enabled.stage and enabled.consistency)

    # The source must not have drifted back to the inheriting form.
    source = (REPO_ROOT / "src" / "checks.py").read_text()
    body = source[source.index("def rhso_direct_terminal_planning"):]
    body = body[:body.index("def rhso_state_regularization")]
    check("checks.py disables the three flags explicitly",
          all(("%s=False" % f) in body for f in
              ("rhso_stage_diagnostics", "rhso_consistency_diagnostics",
               "rhso_jacobian_diagnostics")))
    check("the detail no longer hard-codes an 'unchanged' claim",
          "ALGORITHMIC COUNTERS CHANGED" in body,
          "it now names the counters that differ instead")


# =====================================================================================
# 3+4. BUGS 3 and 4 -- shard-private files and a complete merged per-image CSV
# =====================================================================================
def _fake_run_dir(tmp: Path):
    """A two-shard run directory shaped exactly like the smoke test's output."""
    import run as runpy

    specs = []
    jobs = [("denoising_smoke", "pmf", 0), ("deblurring_smoke", "pmf", 1),
            ("denoising_smoke", "jit", 0), ("deblurring_smoke", "jit", 1)]
    for experiment, model, _ in jobs:
        job_id = "%s_%s_rhso" % (experiment, model)
        specs.append(job_id)
        out = tmp / experiment / model / "rhso" / "t0_1.0"
        out.mkdir(parents=True, exist_ok=True)
        image_ids = ["img%d" % i for i in range(4)]
        payload = {
            "job_id": job_id, "status": "ok", "run_id": "run_1",
            "experiment": experiment, "model": model, "method": "rhso",
            "psnr": 20.0, "ssim": 0.5, "lpips": 0.3,
            "problem_metadata": {"image_ids": image_ids},
            "per_image": {"psnr_per_image": [20.0, 21.0, 22.0, 23.0],
                          "ssim_per_image": [0.5, 0.6, 0.7, 0.8],
                          "lpips_per_image": [0.3, 0.2, 0.1, 0.05]},
        }
        runpy.save_json(out / "metadata.json", payload)
    return specs


def test_shard_files_and_aggregation() -> None:
    print("\n3+4. array shards: private files, complete merged results_per_image.csv")
    import csv

    import run as runpy

    check("shard_suffix separates the tasks",
          runpy.shard_suffix(None) == ""
          and runpy.shard_suffix((0, 2)) == "_shard00"
          and runpy.shard_suffix((1, 2)) == "_shard01")

    canonical = [name for name, _ in runpy.SHARD_FILE_PATTERNS]
    check("checks.json and run_metadata.json are shard-private",
          "checks.json" in canonical and "run_metadata.json" in canonical,
          "the two files the real run lost")
    check("results_per_image.csv is in the shard model",
          "results_per_image.csv" in canonical)

    # run.py must not write the unsuffixed names during a sharded run.
    source = (REPO_ROOT / "run.py").read_text()
    main_body = source[source.index("def main("):]
    for name in ("checks", "run_metadata"):
        check("main() writes %s with the shard suffix" % name,
              ('run_dir / ("%s%%s.json" %% suffix)' % name) in main_body,
              "no unsuffixed concurrent write")

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        job_ids = _fake_run_dir(tmp)

        class FakeSpec:
            def __init__(self, job_id):
                self.job_id = job_id
                self.label = job_id

        class FakePlan:
            specs = [FakeSpec(j) for j in job_ids]

        records, rows, _recons = runpy.collect_finished_jobs(tmp, FakePlan())
        check("every finished job is collected", len(records) == 4,
              "%d job record(s)" % len(records))
        check("4 jobs x 4 images -> exactly 16 per-image rows", len(rows) == 16,
              "%d row(s)" % len(rows))
        check("no duplicate (job_id, image_id) pairs",
              len({(r["job_id"], r["image_id"]) for r in rows}) == 16)
        check("per-image metrics are the per-image values, not the job mean",
              [r["psnr"] for r in rows[:4]] == [20.0, 21.0, 22.0, 23.0])

        # Deterministic ordering: repeated collection is byte-identical.
        again, rows_again, _ = runpy.collect_finished_jobs(tmp, FakePlan())
        check("collection is deterministic",
              [r["job_id"] for r in records] == [r["job_id"] for r in again]
              and [r["image_id"] for r in rows] == [r["image_id"] for r in rows_again])

        # The merged CSV really lands on disk, with the full schema.
        writer = runpy.ResultWriter(tmp)
        writer.records, writer.image_rows = records, rows
        writer.write_csv()
        per_image = tmp / "results_per_image.csv"
        check("aggregation writes results_per_image.csv", per_image.exists())
        with open(per_image) as fh:
            merged = list(csv.DictReader(fh))
        check("results_per_image.csv holds all 16 rows", len(merged) == 16,
              "%d data row(s)" % len(merged))
        check("the per-image schema equals the results schema",
              list(merged[0].keys()) == runpy.RESULT_COLUMNS)

        # Partial completion: a shard that never finished must not break the merge.
        (tmp / "deblurring_smoke" / "jit" / "rhso" / "t0_1.0" / "metadata.json").unlink()
        partial, partial_rows, _ = runpy.collect_finished_jobs(tmp, FakePlan())
        check("a missing shard degrades to a partial merge, not a crash",
              len(partial) == 3 and len(partial_rows) == 12,
              "%d job(s), %d row(s)" % (len(partial), len(partial_rows)))


def test_shard_checks_and_metadata_merge() -> None:
    print("\n3b. merged checks.json contains BOTH model families")
    import run as runpy

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        # Shard 00 ran pMF, shard 01 ran JiT -- the real smoke layout. Both also ran the
        # shared structural checks.
        runpy.save_json(tmp / "checks_shard00.json", [
            {"scope": "structural", "name": "time_grid", "passed": True, "detail": "a"},
            {"scope": "pmf", "name": "rhso_direct_terminal_planning", "passed": True,
             "detail": "pmf ok"},
            {"scope": "pmf", "name": "rhso_receding_horizon", "passed": True, "detail": ""},
        ])
        runpy.save_json(tmp / "checks_shard01.json", [
            {"scope": "structural", "name": "time_grid", "passed": True, "detail": "a"},
            {"scope": "jit", "name": "rhso_direct_terminal_planning", "passed": True,
             "detail": "jit ok"},
            {"scope": "jit", "name": "rhso_receding_horizon", "passed": True, "detail": ""},
        ])
        merged, shards = runpy.merge_shard_checks(tmp)
        scopes = {c["scope"] for c in merged}
        check("both shards are read", shards == [0, 1])
        check("merged checks include pMF AND JiT",
              {"pmf", "jit"} <= scopes, "scopes: %s" % sorted(scopes))
        check("the structural check is de-duplicated",
              len([c for c in merged if c["name"] == "time_grid"]) == 1)
        check("de-duplication records both shards",
              [c for c in merged if c["name"] == "time_grid"][0]["shards"] == [0, 1])
        check("merging is deterministic",
              [(c["scope"], c["name"]) for c in merged]
              == [(c["scope"], c["name"]) for c in runpy.merge_shard_checks(tmp)[0]])

        # A failure in ONE shard must survive the merge.
        runpy.save_json(tmp / "checks_shard01.json", [
            {"scope": "structural", "name": "time_grid", "passed": False,
             "detail": "broke here"},
            {"scope": "jit", "name": "rhso_receding_horizon", "passed": True, "detail": ""},
        ])
        merged, _ = runpy.merge_shard_checks(tmp)
        grid = [c for c in merged if c["name"] == "time_grid"][0]
        check("a failure in any shard survives the merge", grid["passed"] is False,
              "a merge must never be able to hide a failure")

    print("\n3c. merged run_metadata.json keeps every shard's provenance")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        runpy.save_json(tmp / "run_metadata_shard00.json", {
            "run_id": "run_1", "seed": 42, "elapsed_seconds": 100.0,
            "accelerator": {"kind": "gpu", "gpu_name": "H100"},
            "model_provenance": {"pmf": {"sha": "aaa"}},
            "model_load_seconds": {"pmf": 12.0},
            "problems": {"denoising": {"sigma": 0.2}},
            "warnings": ["shared warning", "pmf only"]})
        runpy.save_json(tmp / "run_metadata_shard01.json", {
            "run_id": "run_1", "seed": 42, "elapsed_seconds": 150.0,
            "accelerator": {"kind": "gpu", "gpu_name": "H100"},
            "model_provenance": {"jit": {"sha": "bbb"}},
            "model_load_seconds": {"jit": 30.0},
            "problems": {"deblurring": {"sigma": 0.1}},
            "warnings": ["shared warning", "jit only"]})

        merged = runpy.merge_shard_metadata(tmp, [], [])
        check("both models' provenance survives",
              set(merged["model_provenance"]) == {"pmf", "jit"},
              "not last-writer-wins")
        check("load seconds are unioned",
              set(merged["model_load_seconds"]) == {"pmf", "jit"})
        check("problems from both shards survive",
              set(merged["problems"]) == {"denoising", "deblurring"})
        check("warnings are unioned without duplicates",
              merged["warnings"] == ["shared warning", "pmf only", "jit only"])
        check("elapsed_seconds sums the shards", merged["elapsed_seconds"] == 250.0)
        check("each shard's complete payload is preserved",
              set(merged["shard_metadata"]) == {"00", "01"},
              "no silent lost metadata")
        check("per-shard provenance is listed",
              [s["shard"] for s in merged["shards"]] == [0, 1])
        check("seed is carried through", merged["seed"] == 42)


# =====================================================================================
# 5. BUG 7 -- the stale suffix warning
# =====================================================================================
def test_suffix_warning_only_for_suffix_mode() -> None:
    print("\n5. the REMAINING-suffix warning appears only for suffix-mode jobs")
    from src.config import check_accelerator_compatibility

    from tests.spec_support import build_plan

    def notes_for(model, terminal_mode):
        plan, _warnings = build_plan(
            {"rhso": {"num_rhso_steps": 2, "num_opt_steps": 2,
                      "rhso_terminal_mode": terminal_mode}},
            model=model)
        return check_accelerator_compatibility(plan, {"kind": "cpu"})

    marker = "REMAINING suffix"
    jit_direct = notes_for("jit", "direct")
    check("JiT-direct does NOT claim the suffix is differentiated",
          not any(marker in n for n in jit_direct),
          "this is the stale warning the smoke dry-run printed")
    check("JiT-direct gets an accurate direct-mode note",
          any("direct clean-endpoint prediction" in n for n in jit_direct))

    jit_suffix = notes_for("jit", "suffix")
    check("JiT-suffix DOES keep the warning",
          any(marker in n for n in jit_suffix),
          "genuine suffix jobs must still be warned")

    pmf = notes_for("pmf", "direct")
    check("pMF never gets the suffix warning", not any(marker in n for n in pmf))

    # `auto` resolves per family: JiT -> suffix (legacy), pMF -> direct.
    jit_auto = notes_for("jit", "auto")
    check("JiT with terminal_mode=auto still warns (auto -> suffix)",
          any(marker in n for n in jit_auto),
          "the fix reads the RESOLVED mode, it does not silence the warning")


# =====================================================================================
# 6. BUG 6 -- cluster-conditional CUDA modules
# =====================================================================================
def _shell(snippet: str) -> str:
    result = subprocess.run(
        ["bash", "-c", "source scripts/cluster_modules.sh\n" + snippet],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    return result.stdout.strip()


def test_cluster_module_selection() -> None:
    print("\n6. cluster module selection: Rorqual pinned, Narval untouched")

    rorqual = _shell('mpcflow_cuda_load_line rorqual')
    check("Rorqual pins cuda/12.9", "cuda/12.9" in rorqual)
    check("Rorqual pins cudnn/9.13.1.26", "cudnn/9.13.1.26" in rorqual,
          "the cuDNN that requires cudacore/.12.9.1")
    check("Rorqual still falls back if the pin disappears",
          "|| module load cuda cudnn" in rorqual)

    # The one thing that must not change.
    legacy = ("module load cuda cudnn >/dev/null 2>&1 || "
              "module load cuda >/dev/null 2>&1 || true")
    narval = _shell('mpcflow_cuda_load_line narval')
    check("Narval is byte-identical to the previous behaviour", narval == legacy,
          "no version pin is introduced")
    check("Narval never sees CUDA 12.9", "12.9" not in narval)
    check("an unknown cluster also keeps the default",
          _shell('mpcflow_cuda_load_line some-new-cluster') == legacy)
    check("nibi keeps the default too", _shell('mpcflow_cuda_load_line nibi') == legacy)

    # Both halves of setup_cluster.sh must agree, which they do by construction: the
    # generated activation preamble is produced by the same helper.
    preamble = _shell('mpcflow_module_preamble rorqual 3.11')
    check("the generated Rorqual activation loads 12.9 / 9.13.1.26",
          "cuda/12.9" in preamble and "cudnn/9.13.1.26" in preamble,
          "activate_cluster.sh cannot revert to the old default")
    narval_preamble = _shell('mpcflow_module_preamble narval 3.11')
    check("the generated Narval activation is unpinned",
          "12.9" not in narval_preamble and "module load cuda cudnn" in narval_preamble)

    setup = (REPO_ROOT / "setup_cluster.sh").read_text()
    check("setup_cluster.sh sources the shared helper",
          "scripts/cluster_modules.sh" in setup)
    check("setup_cluster.sh has no hard-coded cuda module load",
          "module load cuda" not in setup,
          "both halves go through mpcflow_cuda_load_line / _preamble")
    check("the venv build uses the resolved line", 'eval "$CUDA_LOAD_LINE"' in setup)
    check("the activation is generated from the same helper",
          "mpcflow_module_preamble" in setup)
    check("an env override is available for future clusters",
          "MPCFLOW_CUDA_MODULE" in
          (REPO_ROOT / "scripts" / "cluster_modules.sh").read_text())
    check("MPCFLOW_CUDA_MODULE overrides the cluster default",
          "cuda/13.2" in _shell('MPCFLOW_CUDA_MODULE=cuda/13.2 '
                                'mpcflow_cuda_load_line narval'))


def test_nvml_dependency_is_declared() -> None:
    print("\n6b. NVML bindings are installed reproducibly")
    setup = (REPO_ROOT / "setup_cluster.sh").read_text()
    check("setup_cluster.sh installs nvidia-ml-py",
          "nvidia-ml-py" in setup)
    check("it prefers the Alliance wheelhouse",
          "pip install --no-index nvidia-ml-py" in setup, "--no-index first")
    check("with a PyPI fallback",
          "|| pip install nvidia-ml-py" in setup.replace("\\\n", ""))
    check("setup reports whether pynvml imports", "pynvml" in setup)
    check("setup says an import is NOT proof NVML works",
          "NVMLError_DriverNotLoaded" in setup and "EXPECTED" in setup,
          "a login node has no driver; that is not a setup failure")
    check("nvidia-ml-py is a declared requirement",
          "nvidia-ml-py" in (REPO_ROOT / "requirements.txt").read_text())


# =====================================================================================
# 7. memory semantics must not regress
# =====================================================================================
def test_memory_never_uses_device_total() -> None:
    print("\n7. process memory is never total device memory")
    import ast

    source = (REPO_ROOT / "src" / "memory.py").read_text()
    tree = ast.parse(source)

    # Look at the CODE, not the text: `nvmlDeviceGetMemoryInfo` is named in a docstring
    # that forbids it, and a substring search cannot tell that apart from a real call.
    # Attribute access covers `nvml.nvmlDeviceGetMemoryInfo(...)`; the string-constant scan
    # covers the `getattr(nvml, "...")` form the module uses for version-dependent APIs.
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    getattr_names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"):
            for arg in node.args[1:]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    getattr_names.add(arg.value)
    # Names built from a list of candidate API strings, as the process query does.
    literal_strings = {node.value for node in ast.walk(tree)
                       if isinstance(node, ast.Constant) and isinstance(node.value, str)
                       and "nvmlDevice" in node.value}
    called = attributes | getattr_names | literal_strings

    check("nvmlDeviceGetMemoryInfo is never called or looked up",
          "nvmlDeviceGetMemoryInfo" not in called,
          "total device memory is not process memory")
    check("the prohibition is still documented",
          "nvmlDeviceGetMemoryInfo" in source,
          "the docstring explaining why remains")
    check("process sampling goes through the per-PID NVML query",
          any(name.startswith("nvmlDeviceGetComputeRunningProcesses")
              for name in called))

    from src.memory import GpuMemoryProfiler

    class NoGpuAdapter:
        class spec:
            framework = "jax"
            name = "pmf"

    profiler = GpuMemoryProfiler(NoGpuAdapter(), enabled=True)
    report = profiler.report()
    process_peak = report.get("gpu_process_peak_gib")
    source_label = report.get("gpu_process_memory_source")
    check("with no NVML the cross-framework field stays unavailable",
          process_peak is None,
          "gpu_process_peak_gib=%r source=%r" % (process_peak, source_label))
    # The label may legitimately explain WHY it is unavailable ("cpu_no_gpu_memory" here).
    # What must never happen is a process peak reported from a framework allocator or from
    # device-total memory, so require that any non-null peak be NVML process sampling.
    label = str(source_label or "")
    check("and it is never silently filled from a framework counter",
          process_peak is None or "nvml_process_sampling" in label,
          "source=%r" % (source_label,))
    check("the unavailability label does not claim NVML sampling",
          "nvml_process_sampling" not in label, "source=%r" % (source_label,))


# =====================================================================================
# 8. the shipped theory configs must keep their scientific design
# =====================================================================================
def test_theory_configs_unchanged() -> None:
    print("\n8. theory configs still resolve to their intended plans")
    import yaml

    from src.config import resolve_run_plan, validate_config

    expected = {
        "experiments_theory_smoke.yaml": 4,
        "experiments_theory_validation_final100.yaml": 70,
        "experiments_theory_jacobian_8img.yaml": 10,
        "experiments_theory_resources_batch4.yaml": 10,
    }
    for name, count in expected.items():
        config = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
        warnings_ = validate_config(config)
        plan = resolve_run_plan(config, warnings_, run_id="probe")
        check("%s -> %d atomic jobs" % (name, count), len(plan.specs) == count,
              "%d resolved" % len(plan.specs))

    # The smoke plan's planner identities are the run's headline invariant.
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "experiments_theory_smoke.yaml").read_text())
    plan = resolve_run_plan(config, validate_config(config), run_id="probe")
    planners = {s.model: s.rhso_terminal_planner for s in plan.specs}
    check("pMF plans with the learned finite-interval map",
          planners.get("pmf") == "learned_finite_interval_map", planners.get("pmf"))
    check("JiT plans with the direct clean-endpoint prediction",
          planners.get("jit") == "direct_clean_endpoint_prediction", planners.get("jit"))
    check("the two planners are NOT reported as one object",
          planners.get("pmf") != planners.get("jit"))

    # Matched-budget design of the main validation config.
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "experiments_theory_validation_final100.yaml").read_text())
    plan = resolve_run_plan(config, validate_config(config), run_id="probe")
    pairs = {(s.num_rhso_steps, s.num_opt_steps) for s in plan.specs}
    for pair in ((1, 160), (2, 80), (4, 40), (8, 20)):
        check("matched budget %s survives" % (pair,), pair in pairs)


def main() -> int:
    print("=" * 94)
    print("Regression tests for the Rorqual smoke-test findings")
    print("=" * 94)
    test_probe_spec_override()
    test_direct_planning_probe_disables_diagnostics()
    test_shard_files_and_aggregation()
    test_shard_checks_and_metadata_merge()
    test_suffix_warning_only_for_suffix_mode()
    test_cluster_module_selection()
    test_nvml_dependency_is_declared()
    test_memory_never_uses_device_total()
    test_theory_configs_unchanged()

    total = PASSED[0] + len(FAILED)
    print("\n%d/%d checks passed" % (PASSED[0], total))
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
