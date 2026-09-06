"""GPU peak-memory measurement for one atomic model-method job.

What is reported, per job (never per image):

    gpu_baseline_gib          steady-state device memory AFTER the model is loaded and
                              warmed up, immediately BEFORE the measured reconstruction
    gpu_peak_gib              the highest device memory observed DURING it
    gpu_incremental_peak_gib  gpu_peak_gib - gpu_baseline_gib, i.e. what the method itself
                              added on top of a resident model
    gpu_memory_source         HOW the number was obtained -- these are not all the same
                              kind of measurement and must not be compared blindly

Memory is a property of the job and its computational batch.  It is deliberately NOT
divided by the batch size: activation memory does not decompose into an honest
"per image" number, and a job that runs batch 4 has not measured four independent images'
worth of anything.  Batch sizes greater than 1 are fully supported; always report the
configured batch size alongside the memory number.

Scope: the profiler wraps ONLY the measured reconstruction, so model loading, dataset
preparation, metrics, visualisation and the untimed warm-up are outside it.  PnP's initial
prior projection and D-Flow's backward pass ARE inside it -- they are part of the method.

Torch
    torch.cuda's own allocator statistics, with the peak counter reset at the boundary.
    This is a true allocator high-water mark for the measured region.

CROSS-FRAMEWORK COMPARISON  (gpu_process_*)
    A Torch allocator high-water mark and a JAX device-memory statistic measure different
    things and must never be put in the same column of a paper table.  Independently of the
    framework-specific numbers above, this module therefore samples ONE comparable metric
    for both frameworks:

        gpu_process_baseline_gib          this PROCESS's GPU memory immediately before the
                                          measured reconstruction
        gpu_process_peak_gib              the highest value sampled DURING it
        gpu_process_incremental_peak_gib  the difference
        gpu_process_memory_source         how it was obtained

    It is sampled through NVML, restricted to this process's own usage where the driver
    exposes per-process accounting, and a FRESH sampler is created for every atomic job --
    unlike JAX's `peak_bytes_in_use`, which is a lifetime high-water mark with no reset API.
    When NVML is unavailable the process fields report `unavailable`; JAX's lifetime peak is
    NEVER substituted for them, because it cannot answer "what did THIS job peak at" once an
    earlier job has peaked higher.  That substitution is exactly what made pMF's memory look
    suspiciously constant across jobs.

JAX
    Deliberately conservative.  This repository sets XLA_PYTHON_CLIENT_PREALLOCATE=false
    and a persistent compilation cache on purpose, and nothing here changes preallocation,
    the allocator, the compilation-cache policy or the device lifecycle to obtain a number.
    JAX exposes `peak_bytes_in_use` per device, but it is a lifetime high-water mark with
    no reset API, so it cannot answer "what did THIS job peak at" once a previous job has
    already peaked higher.  The default is therefore a short NVML sampling loop around the
    synchronised region, clearly labelled as a SAMPLED PROCESS peak -- which is not the
    same thing as an allocator high-water mark, and is reported as such.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

GIB = float(2 ** 30)

# Reported when nothing usable is available, so the columns exist but claim nothing.
UNAVAILABLE = {"gpu_baseline_gib": None, "gpu_peak_gib": None,
               "gpu_incremental_peak_gib": None, "gpu_memory_source": "unavailable"}

# The ONE metric that is comparable between PyTorch and JAX, and the only one a
# cross-framework memory table may use.  See `GpuMemoryProfiler` for why an allocator
# high-water mark and JAX's lifetime peak are not interchangeable with it.
PROCESS_UNAVAILABLE = {"gpu_process_baseline_gib": None, "gpu_process_peak_gib": None,
                       "gpu_process_incremental_peak_gib": None,
                       "gpu_process_memory_source": "unavailable"}


# =====================================================================================
# NVML -- optional, used for the comparable per-process cross-framework metric
# =====================================================================================
def _nvml():
    """Import an NVML binding if one happens to be installed; never a hard requirement.

    pynvml ships with the NVIDIA drivers on most clusters and as a dependency of several
    packages already in `requirements.txt`'s dependency tree, so this usually just works.
    When it does not, memory reporting degrades to "unavailable" and the reconstruction is
    completely unaffected.
    """
    try:
        import pynvml                                                    # type: ignore
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        try:
            from pynvml import smi                                       # type: ignore  # noqa
        except Exception:
            return None
    return None


def _visible_device_token() -> str:
    """Best available identifier for the first GPU exposed to this process.

    Slurm commonly uses a physical numeric index in ``CUDA_VISIBLE_DEVICES``; container
    runtimes may instead expose a full GPU/MIG UUID.  NVML itself uses physical identifiers,
    so preserve the token rather than blindly mapping non-numeric values to device 0.
    """
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        visible = os.environ.get(name, "").strip()
        if not visible or visible.lower() in {"all", "none", "void"}:
            continue
        return visible.split(",")[0].strip()
    return "0"


def _nvml_device_handle(nvml):
    """Resolve the GPU assigned to this process without assuming an index-only CVD."""
    token = _visible_device_token()
    if token.startswith(("GPU-", "MIG-")):
        getter = getattr(nvml, "nvmlDeviceGetHandleByUUID", None)
        if getter is not None:
            # pynvml versions differ on whether UUID arguments are str or bytes.
            for value in (token, token.encode("ascii", errors="ignore")):
                try:
                    return getter(value)
                except Exception:
                    pass
    try:
        return nvml.nvmlDeviceGetHandleByIndex(int(token))
    except Exception:
        # If the token is unusual and cannot be resolved, do not silently choose another
        # GPU.  Returning None makes the process metric explicitly unavailable.
        return None


class _NvmlSampler:
    """Background sampler of THIS process's GPU memory, running only inside the region."""

    def __init__(self, interval: float = 0.02):
        self.interval = max(0.001, float(interval))
        self.peak_bytes: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = _nvml()
        self._handle = None
        if self._nvml is not None:
            try:
                self._handle = _nvml_device_handle(self._nvml)
            except Exception:
                self._handle = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def _used_bytes(self) -> Optional[int]:
        """Return THIS process's resident GPU memory, or ``None`` if NVML cannot.

        Deliberately NEVER falls back to ``nvmlDeviceGetMemoryInfo(...).used``.  That value
        is total DEVICE usage and can include driver allocations and other processes.  If
        per-process accounting is unavailable, the scientifically correct result for the
        ``gpu_process_*`` metric is "unavailable", not a mislabeled device-wide number.
        """
        if self._nvml is None or self._handle is None:
            return None
        try:
            pid = os.getpid()

            # Different pynvml / driver combinations expose different process-query
            # versions.  Try the newest first, then gracefully fall back to older APIs.
            getters = (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetComputeRunningProcesses",
            )
            queried = False
            procs = []
            for getter in getters:
                fn = getattr(self._nvml, getter, None)
                if fn is None:
                    continue
                try:
                    procs = fn(self._handle)
                    queried = True
                    break
                except Exception:
                    continue

            if not queried:
                return None

            # NVML may use its UINT64 sentinel (all bits set) when per-process memory is
            # not available, notably on some platforms / MIG configurations.  Reject any
            # such value rather than turning it into an absurdly large memory reading.
            nvml_na = getattr(self._nvml, "NVML_VALUE_NOT_AVAILABLE", None)
            for proc in procs:
                if int(getattr(proc, "pid", -1)) != pid:
                    continue
                used = getattr(proc, "usedGpuMemory", None)
                if used is None:
                    return None
                try:
                    used_i = int(used)
                except (TypeError, ValueError, OverflowError):
                    return None
                if nvml_na is not None:
                    try:
                        if used_i == int(nvml_na):
                            return None
                    except (TypeError, ValueError, OverflowError):
                        pass
                # Defensive handling for the common UINT64_MAX sentinel even when the
                # binding does not expose NVML_VALUE_NOT_AVAILABLE.
                if used_i < 0 or used_i >= (1 << 63):
                    return None
                return used_i

            # The GPU is visible, but NVML did not return a per-process entry for us.
            # This is NOT permission to use device-wide memory.
            return None
        except Exception:
            return None

    def sample_once(self) -> Optional[float]:
        if not self.available:
            return None
        used = self._used_bytes()
        return None if used is None else used / GIB

    def _loop(self) -> None:
        while not self._stop.is_set():
            used = self._used_bytes()
            if used is not None:
                self.peak_bytes = used if self.peak_bytes is None else max(self.peak_bytes,
                                                                          used)
            self._stop.wait(self.interval)

    def start(self) -> None:
        if not self.available:
            return
        self.peak_bytes = self._used_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="nvml-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> Optional[float]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        # One final synchronous sample, so a very short region is never missed entirely.
        used = self._used_bytes()
        if used is not None:
            self.peak_bytes = used if self.peak_bytes is None else max(self.peak_bytes, used)
        return None if self.peak_bytes is None else self.peak_bytes / GIB


# =====================================================================================
# The profiler
# =====================================================================================
class GpuMemoryProfiler:
    """Measure peak GPU memory across the measured region(s) of ONE atomic job.

    Usage:

        profiler = GpuMemoryProfiler(adapter, enabled=True)
        profiler.establish_baseline()
        with profiler.measure():
            ...the reconstruction...
        record.update(profiler.report())

    `measure()` may be entered several times (one job reconstructs several batch chunks);
    the reported peak is the maximum over them and the baseline is the one taken before the
    first.  Any failure inside the instrumentation degrades to "unavailable" rather than
    breaking the reconstruction.
    """

    def __init__(self, adapter: Any, enabled: bool = True, nvml_interval: float = 0.02):
        self.adapter = adapter
        self.enabled = bool(enabled)
        self.nvml_interval = float(nvml_interval)
        self.framework = getattr(getattr(adapter, "spec", None), "framework", None)
        self.baseline_gib: Optional[float] = None
        self.peak_gib: Optional[float] = None
        self.source: str = "disabled" if not enabled else "uninitialised"
        self.extra: Dict[str, Any] = {}
        self._sampler: Optional[_NvmlSampler] = None
        # The comparable cross-framework metric, measured for BOTH frameworks and reset per
        # job because this object is constructed per atomic job.
        self.process_baseline_gib: Optional[float] = None
        self.process_peak_gib: Optional[float] = None
        self.process_source: str = "disabled" if not enabled else "uninitialised"
        self._process_sampler: Optional[_NvmlSampler] = None
        self._mode = self._choose_mode() if enabled else "off"
        if self._mode not in ("off", "off_cpu"):
            self._process_sampler = (self._sampler if self._mode == "nvml"
                                     else self.make_process_sampler())
            if self._process_sampler is None or not self._process_sampler.available:
                self._process_sampler = None
                self.process_source = "unavailable"
        else:
            self.process_source = ("cpu_no_gpu_memory" if self._mode == "off_cpu"
                                   else self.process_source)

    def make_process_sampler(self) -> Optional["_NvmlSampler"]:
        """A FRESH NVML sampler for this job.  Overridable, so tests can inject a fake."""
        sampler = _NvmlSampler(self.nvml_interval)
        return sampler if sampler.available else None

    # ---------------------------------------------------------------- setup
    def _choose_mode(self) -> str:
        if self.framework == "torch":
            try:
                import torch
                if torch.cuda.is_available():
                    return "torch_cuda"
                return "off_cpu"
            except Exception:
                return "off"
        if self.framework == "jax":
            try:
                import jax
                if jax.default_backend() != "gpu":
                    return "off_cpu"
            except Exception:
                return "off"
            sampler = _NvmlSampler(self.nvml_interval)
            if sampler.available:
                self._sampler = sampler
                return "nvml"
            return "jax_device_stats"
        return "off"

    def _sync(self) -> None:
        """Framework-neutral barrier, so an asynchronous dispatch cannot escape the region."""
        try:
            if self.framework == "torch":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            elif self.framework == "jax":
                import jax
                for device in jax.local_devices():
                    try:
                        device.synchronize_all_activity()
                    except Exception:
                        pass
        except Exception:
            pass

    def _jax_stats(self) -> Dict[str, Any]:
        try:
            import jax
            stats = jax.local_devices()[0].memory_stats() or {}
            return stats
        except Exception:
            return {}

    # ---------------------------------------------------------------- measurement
    def establish_baseline(self) -> Optional[float]:
        """Steady state after loading and warm-up, immediately before the measured work."""
        if self._mode in ("off", "off_cpu") and self._process_sampler is None:
            # Three different things, and a reader of results.csv needs to tell them apart:
            # the user disabled it, the run has no GPU at all, or we genuinely could not
            # measure.  None of them is allowed to look like a measured zero.
            if self._mode == "off_cpu":
                self.source = "cpu_no_gpu_memory"
            elif not self.enabled:
                self.source = "disabled"
            else:
                self.source = "unavailable"
            return None
        self._sync()
        if self._process_sampler is not None:
            self.process_baseline_gib = self._process_sampler.sample_once()
            self.process_source = (("nvml_process_sampling(interval=%gs)"
                                    % self.nvml_interval)
                                   if self.process_baseline_gib is not None
                                   else "unavailable")
        try:
            if self._mode == "torch_cuda":
                import torch
                self.baseline_gib = torch.cuda.memory_allocated() / GIB
                self.extra["gpu_baseline_reserved_gib"] = torch.cuda.memory_reserved() / GIB
                torch.cuda.reset_peak_memory_stats()
                self.source = "torch.cuda.max_memory_allocated"
            elif self._mode == "nvml":
                self.baseline_gib = self._sampler.sample_once()
                stats = self._jax_stats()
                if stats.get("bytes_in_use") is not None:
                    self.extra["gpu_baseline_jax_bytes_in_use_gib"] = \
                        float(stats["bytes_in_use"]) / GIB

                if self.baseline_gib is not None:
                    self.source = (
                        "nvml_process_sampling(interval=%gs)  [SAMPLED PROCESS PEAK, "
                        "not an allocator high-water mark]" % self.nvml_interval
                    )
                elif stats:
                    # NVML can see the device but cannot provide PID-specific accounting.
                    # Keep the cross-framework gpu_process_* metric unavailable, while still
                    # exposing JAX's framework-specific lifetime statistic in its historical
                    # gpu_* columns with an unmistakable label.
                    self._mode = "jax_device_stats"
                    self.baseline_gib = float(stats.get("bytes_in_use", 0)) / GIB
                    self.extra["gpu_lifetime_peak_at_baseline_gib"] = \
                        float(stats.get("peak_bytes_in_use", 0)) / GIB
                    self.source = (
                        "jax_device_memory_stats(peak_bytes_in_use)  [LIFETIME "
                        "high-water mark: it has no reset API, so it can only "
                        "over-report this job]"
                    )
                else:
                    self.source = "unavailable"
                    self._mode = "off"
                    self.baseline_gib = None
            else:                                                # jax_device_stats
                stats = self._jax_stats()
                if not stats:
                    self.source = "unavailable"
                    self._mode = "off"
                    return None
                self.baseline_gib = float(stats.get("bytes_in_use", 0)) / GIB
                self.extra["gpu_lifetime_peak_at_baseline_gib"] = \
                    float(stats.get("peak_bytes_in_use", 0)) / GIB
                self.source = ("jax_device_memory_stats(peak_bytes_in_use)  [LIFETIME "
                               "high-water mark: it has no reset API, so it can only "
                               "over-report this job]")
        except Exception as exc:                                         # pragma: no cover
            self.extra["gpu_memory_error"] = "%s: %s" % (type(exc).__name__, exc)
            self.source = "unavailable"
            self._mode = "off"
            self.baseline_gib = None
        return self.baseline_gib

    class _Region:
        def __init__(self, profiler: "GpuMemoryProfiler"):
            self.profiler = profiler

        def __enter__(self):
            self.profiler._enter_region()
            return self.profiler

        def __exit__(self, *exc):
            self.profiler._exit_region()
            return False

    def measure(self) -> "GpuMemoryProfiler._Region":
        return GpuMemoryProfiler._Region(self)

    def _enter_region(self) -> None:
        # The comparable process metric is deliberately INDEPENDENT of the
        # framework-specific one: if JAX's device statistics are unavailable, or the Torch
        # allocator path is off, NVML process sampling still answers "what did this job
        # peak at" and must not be switched off with it.
        if self._mode in ("off", "off_cpu") and self._process_sampler is None:
            return
        try:
            self._sync()
            if self._process_sampler is not None:
                self._process_sampler.start()
            elif self._mode == "nvml" and self._sampler is not None:
                self._sampler.start()
        except Exception:                                                # pragma: no cover
            pass

    def _exit_region(self) -> None:
        if self._mode in ("off", "off_cpu") and self._process_sampler is None:
            return
        try:
            self._sync()
            if self._process_sampler is not None:
                sampled = self._process_sampler.stop()
                if sampled is not None:
                    self.process_peak_gib = (sampled if self.process_peak_gib is None
                                             else max(self.process_peak_gib, sampled))
            observed: Optional[float] = None
            if self._mode in ("off", "off_cpu"):
                pass                       # process sampling only; nothing else to read
            elif self._mode == "torch_cuda":
                import torch
                observed = torch.cuda.max_memory_allocated() / GIB
                self.extra["gpu_peak_reserved_gib"] = max(
                    float(self.extra.get("gpu_peak_reserved_gib") or 0.0),
                    torch.cuda.max_memory_reserved() / GIB)
            elif self._mode == "nvml" and self._sampler is not None:
                # The same sampler object serves both metrics on the JAX path; it was
                # already stopped above, so its peak is read rather than re-stopped.
                observed = (None if self.process_peak_gib is None
                            else self.process_peak_gib)
            elif self._mode == "jax_device_stats":
                stats = self._jax_stats()
                if stats:
                    observed = float(stats.get("peak_bytes_in_use", 0)) / GIB
            if observed is not None:
                self.peak_gib = observed if self.peak_gib is None else max(self.peak_gib,
                                                                          observed)
        except Exception as exc:                                         # pragma: no cover
            self.extra["gpu_memory_error"] = "%s: %s" % (type(exc).__name__, exc)

    # ---------------------------------------------------------------- output
    def process_report(self) -> Dict[str, Any]:
        """The cross-framework process metric, or an explicit `unavailable`.

        Never falls back to JAX's `peak_bytes_in_use`: that is a LIFETIME high-water mark
        with no reset API, so once an earlier job has peaked higher it reports that earlier
        job's peak rather than this one's. Reporting it here would be the very mislabelling
        that made pMF's per-job memory look constant.
        """
        if self.process_peak_gib is None or self.process_baseline_gib is None:
            out = dict(PROCESS_UNAVAILABLE)
            out["gpu_process_memory_source"] = self.process_source
            return out
        incremental = self.process_peak_gib - self.process_baseline_gib
        return {
            "gpu_process_baseline_gib": round(float(self.process_baseline_gib), 4),
            "gpu_process_peak_gib": round(float(self.process_peak_gib), 4),
            # A sampled process peak can land marginally below the baseline sample; report
            # the clamp rather than a negative "incremental" number.
            "gpu_process_incremental_peak_gib": round(float(max(incremental, 0.0)), 4),
            "gpu_process_memory_source": self.process_source,
        }

    def report(self) -> Dict[str, Any]:
        """Framework-specific numbers PLUS the comparable process numbers.

        Both sets are always present and separately labelled. The historical
        `gpu_*` columns keep exactly the meaning they had; the `gpu_process_*` columns are
        the ones a JiT-vs-pMF table may compare.
        """
        if self.peak_gib is None or self.baseline_gib is None:
            out = dict(UNAVAILABLE)
            out["gpu_memory_source"] = self.source
            out.update(self.extra)
            out.update(self.process_report())
            return out
        incremental = self.peak_gib - self.baseline_gib
        out = {
            "gpu_baseline_gib": round(float(self.baseline_gib), 4),
            "gpu_peak_gib": round(float(self.peak_gib), 4),
            # A sampled process peak can land marginally below the baseline sample; report
            # the clamp rather than a negative "incremental" number.
            "gpu_incremental_peak_gib": round(float(max(incremental, 0.0)), 4),
            "gpu_memory_source": self.source,
        }
        out.update({k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in self.extra.items()})
        out.update(self.process_report())
        return out
