"""pixel MeanFlow (pMF) adapter -- class-conditional MeanFlows directly in pixel space.

Carried over from both notebooks: config overrides, EMA-key selection (pMF stores
`ema_params` as a DICT keyed by EMA value, unlike iMF's single pytree), the `noise_scale`
that `generate()` applies to its Gaussian, and the native `sample_one_step` API.

pMF's native space IS the canonical pixel representation, so `encode_pixels` and
`to_pixels` are the identity and the differentiable path costs nothing.

pMF is a MeanFlow model: its dynamics are a learned FINITE-INTERVAL transition
T_theta(x; s -> r), not an instantaneous velocity field.  It is never approximated as an
ordinary flow model anywhere in this repository.  The reversed native clock t_MF = 1 - s
lives in `transition()` and nowhere else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..utils import (MEANFLOW, assert_pixel_batch, gaussian_noise, native_time, record_time)
from .base import (AdapterSpec, Conditioning, MeanFlowAdapter, RepoSandbox,
                   download_and_extract_zip, find_checkpoint_dir, register_adapter)


class PMFAdapter(MeanFlowAdapter):

    def __init__(self, registry: Dict[str, Any], repo_dir: Path, ckpt_cache: Path):
        import jax
        import jax.numpy as jnp
        super().__init__("pmf", registry)
        self.jax, self.jnp = jax, jnp
        # Resolved BEFORE the sandbox chdirs into the repository: every path used
        # inside `with self.sandbox:` must be absolute.
        self.repo_dir = Path(repo_dir).resolve()
        self.ckpt_cache = Path(ckpt_cache).resolve()
        self.sandbox = RepoSandbox("pmf", self.repo_dir, extra_roots=("pmf",))
        self._load()

    def _load(self) -> None:
        import time
        jax, jnp = self.jax, self.jnp
        t_start = time.perf_counter()
        rcfg = self.registry
        with self.sandbox:
            import yaml
            from configs.default import get_config as get_default_config
            from pmf import pixelMeanFlow
            from utils.ckpt_util import restore_checkpoint
            from utils.trainstate_util import create_train_state
            from utils.lr_utils import lr_schedules

            cfg = get_default_config()
            with open(self.repo_dir / rcfg["config_yml"]) as fh:
                overrides = yaml.safe_load(fh)
            for k, v in overrides.items():
                cfg[k].update(v) if isinstance(v, dict) else cfg.__setitem__(k, v)

            # Eval-only: disable auxiliary training losses (preserves the source config).
            cfg.model.lpips = False
            cfg.model.convnext = False
            cfg.model.lpips_lambda = 0.0
            cfg.model.convnext_lambda = 0.0
            cfg.eval_only = True
            cfg.load_from = ""
            cfg.training.ema_val = [500, 1000, 2000]
            cfg.training.ema_type = "edm"
            cfg.logging.use_wandb = False
            cfg.logging.wandb_project = ""
            cfg.logging.wandb_entity = ""
            cfg.logging.wandb_notes = ""
            cfg.logging.wandb_tags = []
            guid = rcfg["guidance"]
            cfg.sampling.num_steps = 1
            cfg.sampling.omega = guid["scale"]
            cfg.sampling.t_min = guid["interval"][0]
            cfg.sampling.t_max = guid["interval"][1]
            cfg.sampling.emas = [1000]
            cfg.sampling.interval = [list(guid["interval"])]
            cfg.sampling.omegas = [guid["scale"]]

            extracted = download_and_extract_zip(rcfg["hf_repo"], rcfg["ckpt_file"],
                                                 self.ckpt_cache)
            cfg.load_from = find_checkpoint_dir(extracted)

            self.config = cfg
            self.image_size = int(cfg.dataset.image_size)          # 256
            self.image_channels = int(cfg.dataset.image_channels)  # 3
            self.num_classes = int(cfg.dataset.num_classes)        # 1000

            self._model = pixelMeanFlow(
                model_str=cfg.model.model_str, num_classes=cfg.model.num_classes,
                P_mean=cfg.model.P_mean, P_std=cfg.model.P_std,
                noise_scale=cfg.model.noise_scale,
                data_proportion=cfg.model.data_proportion, cfg_beta=cfg.model.cfg_beta,
                cfg_max=cfg.model.cfg_max, class_dropout_prob=cfg.model.class_dropout_prob,
                norm_p=cfg.model.norm_p, norm_eps=cfg.model.norm_eps, eval=True)
            state = create_train_state(jax.random.key(0), cfg, self._model,
                                       cfg.dataset.image_size, lr_schedules(cfg, 1000))
            state = restore_checkpoint(state, cfg.load_from)
            self.step = int(state.step)

            available = list(state.ema_params.keys())
            wanted = rcfg.get("ema_key")
            self.ema_key = wanted if (wanted is not None and wanted in available) \
                else available[-1]
            if wanted is not None and wanted not in available:
                print("   Requested EMA key %r unavailable; using %r (available: %s)"
                      % (wanted, self.ema_key, available))
            self._params = {"params": state.ema_params[self.ema_key]}
            self.noise_scale = float(rcfg["noise_scale"] if rcfg.get("noise_scale") is not None
                                     else cfg.model.noise_scale)
            del state

        self._build_jitted()
        load_s = time.perf_counter() - t_start
        record_time("load/pmf", load_s)

        self.spec = AdapterSpec(
            name="pmf", display_name=rcfg["display_name"], dynamics_family=MEANFLOW,
            framework="jax", state_space="pixel",
            native_shape=(self.image_size, self.image_size, self.image_channels),
            layout="BHWC", pixel_resolution=self.image_size,
            prediction_kind="mean_velocity", native_time_mapping=rcfg["native_time_mapping"],
            batch_size=int(rcfg["batch_size"]), fixed_batch_shape=True,
            num_classes=self.num_classes, null_label=int(rcfg["null_label"]),
            guidance=dict(rcfg["guidance"]),
            checkpoint={"hf_repo": rcfg["hf_repo"], "file": rcfg["ckpt_file"],
                        "ckpt_dir": self.config.load_from, "step": self.step,
                        "model_str": self.config.model.model_str, "ema_key": self.ema_key,
                        "repo_rev": rcfg.get("repo_rev"), "noise_scale": self.noise_scale,
                        "load_seconds": round(load_s, 2)})

    def _build_jitted(self) -> None:
        jax, jnp = self.jax, self.jnp
        model = self._model

        def _step(params, z, labels, t_steps, omega, t_min, t_max):
            # t_steps = [t, r]; i = 0 -> t = t_steps[0] (from), r = t_steps[1] (to).
            # t_steps is a traced ARRAY, so different intervals reuse one compilation.
            return model.apply(params, z, labels, jnp.int32(0), t_steps,
                               omega, t_min, t_max, method=model.sample_one_step)

        self._jit_step = jax.jit(_step)

    # ------------------------------------------------------------------ canonical interface
    def to_native_noise(self, noise: np.ndarray):
        # pMF's generate() scales its Gaussian by model.noise_scale; preserved so the prior
        # epsilon is exactly the model's own prior.
        return self.jnp.asarray(np.asarray(noise, np.float32)) * self.jnp.float32(
            self.noise_scale)

    def encode_pixels(self, pixels: np.ndarray):
        """Identity: pMF's native space IS the canonical pixel space."""
        assert_pixel_batch(pixels, "pmf guide", self.image_size)
        return self.jnp.asarray(np.asarray(pixels, np.float32))

    def _lerp(self, guide, noise, keep: float, add: float):
        return self.jnp.float32(keep) * guide + self.jnp.float32(add) * noise

    def to_pixels(self, state, differentiable: bool = False):
        if differentiable:
            return state
        return np.clip(np.asarray(self.jax.device_get(state), np.float32), -1.0, 1.0)

    # ------------------------------------------------------------------ MeanFlow dynamics
    def transition(self, state, s_from: float, s_to: float, conditioning: Conditioning):
        """T_theta(x; s_from -> s_to) on the canonical clock.

        The reversed native clock t_MF = 1 - s lives here and nowhere else: SDEdit and both
        MPC solvers only ever speak s.  Guidance stays in the checkpoint's native convention
        (omega, [t_min, t_max] in MeanFlow time) and is fused inside one network call.
        """
        jnp = self.jnp
        g = {**self.spec.guidance, **(conditioning.guidance or {})}
        interval = g.get("interval") or self.spec.guidance["interval"]
        labels = jnp.asarray(np.asarray(conditioning.labels, np.int32))
        t_steps = jnp.asarray([native_time(s_from, self.spec.native_time_mapping),
                               native_time(s_to, self.spec.native_time_mapping)], jnp.float32)
        self.count_forwards(1)
        return self._jit_step(self._params, state, labels, t_steps,
                              jnp.float32(g["scale"]),
                              jnp.float32(interval[0]), jnp.float32(interval[1]))

    # ------------------------------------------------------------------ sanity checks
    def sanity_checks(self) -> Dict[str, Any]:
        return {"identity_pixels": self._check_identity,
                "interval_ordering": self._check_interval_ordering,
                "noise_scaling": self._check_noise_scaling}

    def _check_identity(self, ctx) -> Tuple[bool, str]:
        z = self.encode_pixels(ctx["pixels"])
        back = self.to_pixels(z)
        diff_ok = np.allclose(back, ctx["pixels"], atol=1e-6)
        same_obj = self.to_pixels(z, differentiable=True) is z
        return bool(diff_ok and same_obj), \
            "native pixels are canonical pixels; the differentiable path is the identity"

    def _check_interval_ordering(self, ctx) -> Tuple[bool, str]:
        """s_from == s_to must be a no-op, because t - r == 0 in the native step."""
        jnp = self.jnp
        z = jnp.asarray(ctx["pixels"], jnp.float32)
        cond = ctx["conditioning"]
        same = self.block(self.transition(z, 0.5, 0.5, cond))
        moved = self.block(self.transition(z, 0.3, 0.6, cond))
        ok = (bool(jnp.allclose(same, z, atol=1e-4))
              and tuple(moved.shape) == tuple(z.shape) and bool(jnp.isfinite(moved).all()))
        return ok, "s->s is the identity; s=0.3->0.6 gives a finite state of the same shape"

    def _check_noise_scaling(self, ctx) -> Tuple[bool, str]:
        raw = gaussian_noise(self.native_batch_shape(2), "pmf", "check", "noise")
        native = np.asarray(self.to_native_noise(raw))
        ok = np.allclose(native, raw * self.noise_scale, atol=1e-6)
        return ok, "noise_scale=%.3f applied as in generate()" % self.noise_scale


@register_adapter("pmf")
def _make_pmf(registry: Dict[str, Any], context: Dict[str, Any]) -> PMFAdapter:
    return PMFAdapter(registry, context["repo_paths"]["pmf"], context["ckpt_cache"])
