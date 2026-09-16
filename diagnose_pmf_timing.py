#!/usr/bin/env python
"""Diagnostic: is pMF's per-interval cost compilation or computation?

    python diagnose_pmf_timing.py

Loads pMF once, then times the SAME transition twice and a NEW interval twice, reporting
`jax.jit`'s compilation-cache size after each.  If the cache grows when the interval value
changes, compilation is interval-specific and every distinct time grid must be warmed up
separately.  If it does not grow, the cost is genuine computation.

Delete this file once you have your answer; it is not part of the benchmark.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, resolve_run_plan, validate_config
from src.models import ModelManager, load_adapters
from src.models.base import Conditioning
from src.utils import load_dotenv


def cache_size(fn):
    for attr in ("_cache_size", "cache_size"):
        if hasattr(fn, attr):
            try:
                return getattr(fn, attr)()
            except Exception:
                pass
    return "n/a"


def main():
    load_dotenv(".env")
    config = load_config("configs/experiments.yaml")
    plan = resolve_run_plan(config, validate_config(config))
    cache_root = Path(plan.cache_root).resolve()

    import run as runner
    context = runner.ensure_repositories(plan, cache_root, verbose=False)
    context.update(runner.init_frameworks(plan, config, {"kind": "gpu"}, cache_root,
                                          verbose=False))
    context["ckpt_cache"] = (cache_root / "checkpoints").resolve()
    load_adapters(["pmf"])

    manager = ModelManager(config, plan, False, context)
    adapter = manager.acquire("pmf")

    import jax
    x = adapter.prior_sample(["a", "b"])
    cond = Conditioning(labels=np.array([0, 39], np.int32), guidance=dict(adapter.spec.guidance))

    def timed(s_from, s_to):
        started = time.perf_counter()
        jax.block_until_ready(adapter.transition(x, s_from, s_to, cond))
        return time.perf_counter() - started

    print("\n%-28s %10s   %s" % ("call", "seconds", "jit cache entries"))
    print("-" * 62)
    for label, a, b in [("0.2 -> 1.0  (1st)", 0.2, 1.0), ("0.2 -> 1.0  (2nd)", 0.2, 1.0),
                        ("0.2 -> 0.6  (1st, NEW)", 0.2, 0.6),
                        ("0.2 -> 0.6  (2nd)", 0.2, 0.6),
                        ("0.6 -> 1.0  (1st, NEW)", 0.6, 1.0),
                        ("0.6 -> 1.0  (2nd)", 0.6, 1.0)]:
        print("%-28s %10.3f   %s" % (label, timed(a, b), cache_size(adapter._jit_step)))

    print("\nReading:")
    print("  cache entries GROW on a new interval -> compilation is interval-specific;")
    print("     the warm-up must trace every distinct time grid (this is what run.py now does).")
    print("  cache entries CONSTANT but 1st call slow -> one-off host->device parameter")
    print("     transfer; the fix is the same (warm up before timing).")
    print("  every call equally slow -> genuine computation, and pMF really is slower.")
    manager.release_all()


if __name__ == "__main__":
    main()
