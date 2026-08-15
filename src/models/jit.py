"""JiT adapter -- pixel-space clean-image predictor (BCHW 3x256x256), velocity DERIVED.

Carried over from both notebooks (they agreed): checkpoint discovery, the Diffusers-mirror
import dance, the safe dtype policy (BF16 where supported, else FP32, never FP16), the FP32
integration state, `t_eps`, and the interval-gated CFG rule.

JiT predicts x_1, not velocity.  Standard-flow sampling and MPC both need v_theta, so the
adapter derives it with JiT's own relation

        v = (x1_hat - x_t) / max(1 - t, t_eps)

which is exactly what the official sampler integrates.  CFG is applied to the CLEAN
prediction first (its native output), then the velocity is derived -- never the other way
round.

AUTOGRAD stays enabled here.  The SDEdit notebook wrapped prediction in
`torch.inference_mode()` because sampling is pure inference; MPC is not, and inference-mode
tensors can never enter an autograd graph.  Model PARAMETERS keep `requires_grad_(False)`,
so gradients reach the state and the control only and no parameter gradients are allocated.
SDEdit callers get the same saving by wrapping their own loop in `torch.no_grad()`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..utils import (FLOW_ASCENDING, STANDARD_FLOW, assert_pixel_batch, load_python_module,
                     native_time, record_time)
from .base import AdapterSpec, Conditioning, StandardFlowAdapter, register_adapter


class JiTAdapter(StandardFlowAdapter):

    def __init__(self, registry: Dict[str, Any], repo_dir: Path, cache_dir: Path, device, dtype):
        import torch
        super().__init__("jit", registry)
        self.torch = torch
        self.repo_dir = Path(repo_dir)
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.dtype = dtype
        self.integration_dtype = torch.float32
        self.t_eps = float(registry["t_eps"])
        self.noise_scale = float(registry["noise_scale"])
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        import time
        t_start = time.perf_counter()
        cfg = self.registry
        variant = cfg["variant"]
        if cfg["checkpoint_backend"] == "hf_mirror":
            model, source = self._load_hf_mirror(variant)
        else:
            model, source = self._load_original_local(variant)
        self._model = model
        self.backend_kind = cfg["checkpoint_backend"]

        load_s = time.perf_counter() - t_start
        record_time("load/jit", load_s)
        n_params = sum(p.numel() for p in model.parameters())
        if n_params == 0:
            raise RuntimeError("The loaded JiT transformer has no parameters.")

        guidance = dict(cfg["guidance"])
        if guidance.get("scale") is None:
            guidance["scale"] = cfg["recommended_cfg"][variant]
            guidance["scale_source"] = "model_size_recommendation"

        self.spec = AdapterSpec(
            name="jit", display_name=cfg["display_name"], dynamics_family=STANDARD_FLOW,
            framework="torch", state_space="pixel", native_shape=tuple(cfg["native_shape"]),
            layout="BCHW", pixel_resolution=cfg["pixel_resolution"],
            prediction_kind="clean", native_time_mapping=cfg["native_time_mapping"],
            batch_size=int(cfg["batch_size"]), fixed_batch_shape=False,
            num_classes=int(cfg["num_classes"]), null_label=int(cfg["null_label"]),
            guidance=guidance,
            euler_final_step_for_heun=bool(cfg.get("euler_final_step_for_heun", True)),
            checkpoint=dict(source, variant=variant, network_dtype=str(self.dtype),
                            integration_dtype="float32", t_eps=self.t_eps,
                            noise_scale=self.noise_scale,
                            euler_final_step_for_heun=bool(
                                cfg.get("euler_final_step_for_heun", True)),
                            parameters=int(n_params), load_seconds=round(load_s, 2)))

    def _load_hf_mirror(self, variant: str):
        """Load the converted transformer component directly, bypassing DiffusionPipeline."""
        import importlib.util
        from huggingface_hub import snapshot_download

        folder = variant.replace("/", "-")
        repo_id = self.registry["hf_mirror_repo"]
        local_root = self.cache_dir / "jit_diffusers"
        snapshot_download(repo_id=repo_id, allow_patterns=["%s/*" % folder, "%s/**" % folder],
                          local_dir=str(local_root))
        variant_dir = local_root / folder
        transformer_dir = variant_dir / "transformer"
        if not transformer_dir.is_dir():
            raise FileNotFoundError("No transformer/ directory under %s" % variant_dir)

        config_path = transformer_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        component = json.loads(config_path.read_text())
        class_name = component.get("_class_name")
        if not class_name:
            raise RuntimeError("transformer/config.json has no _class_name.")

        # The mirror ships its modelling code alongside the weights; import it under a
        # private name so it cannot collide with anything else in the process.
        module_file = None
        for candidate in sorted(transformer_dir.glob("*.py")):
            if ("class %s" % class_name) in candidate.read_text():
                module_file = candidate
                break
        if module_file is None:
            raise FileNotFoundError("No module under %s defines %s"
                                    % (transformer_dir, class_name))

        full_name = "jit_mirror_" + module_file.stem
        added = False
        try:
            if str(transformer_dir) not in sys.path:
                sys.path.insert(0, str(transformer_dir))
                added = True
            spec = importlib.util.spec_from_file_location(full_name, str(module_file))
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
        finally:
            if added and str(transformer_dir) in sys.path:
                sys.path.remove(str(transformer_dir))

        if not hasattr(module, class_name):
            available = [n for n, v in vars(module).items() if isinstance(v, type)]
            raise AttributeError("%s does not define %r. Available: %s"
                                 % (module_file.name, class_name, available))
        cls = getattr(module, class_name)
        try:
            transformer = cls.from_pretrained(str(transformer_dir), torch_dtype=self.dtype,
                                              local_files_only=True)
        except TypeError as error:                     # diffusers renamed torch_dtype -> dtype
            if "torch_dtype" not in str(error):
                raise
            transformer = cls.from_pretrained(str(transformer_dir), dtype=self.dtype,
                                              local_files_only=True)
        transformer.eval().requires_grad_(False)
        transformer.to(device=self.device, dtype=self.dtype)
        return transformer, {"official_code_repository": self.registry["repo_url"],
                             "checkpoint_mirror": repo_id, "checkpoint_subfolder": folder,
                             "checkpoint_format": "Diffusers/safetensors",
                             "loaded_component": "transformer_only",
                             "conversion": "third-party Diffusers conversion of EMA1",
                             "pipeline_bypassed": True}

    def _load_original_local(self, variant: str):
        torch = self.torch
        path = Path(self.registry["local_checkpoint"])
        if not path.exists():
            raise FileNotFoundError(path)
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        jit_models = load_python_module("jit_official_models", self.repo_dir / "model_jit.py")
        proj_dropout = 0.2 if variant.startswith("JiT-H") else 0.0
        model = jit_models.JiT_models[variant](input_size=256, in_channels=3, num_classes=1000,
                                               attn_drop=0.0, proj_drop=proj_dropout)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_ema1" in checkpoint:
            state_dict, selected = checkpoint["model_ema1"], "model_ema1"
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict, selected = checkpoint["model"], "model"
        else:
            state_dict, selected = checkpoint, "raw_state_dict"
        for prefix in ("module.", "net."):
            keys = list(state_dict.keys())
            if keys and sum(k.startswith(prefix) for k in keys) > len(keys) / 2:
                state_dict = {(k[len(prefix):] if k.startswith(prefix) else k): v
                              for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError("JiT checkpoint mismatch. missing=%s unexpected=%s"
                               % (missing, unexpected))
        model.eval().requires_grad_(False)
        model.to(device=self.device, dtype=self.dtype)
        return model, {"official_code_repository": self.registry["repo_url"],
                       "checkpoint_source": str(path), "selected_weights": selected,
                       "conversion": None}

    # ------------------------------------------------------------------ canonical interface
    def to_native_noise(self, noise: np.ndarray):
        return (self.torch.from_numpy(np.ascontiguousarray(noise))
                .to(device=self.device, dtype=self.integration_dtype) * self.noise_scale)

    def encode_pixels(self, pixels: np.ndarray):
        """Canonical (N,256,256,3) BHWC [-1,1] -> native (N,3,256,256) BCHW [-1,1]."""
        assert_pixel_batch(pixels, "jit guide", self.spec.pixel_resolution)
        arr = np.ascontiguousarray(np.asarray(pixels, np.float32).transpose(0, 3, 1, 2))
        return self.torch.from_numpy(arr).to(device=self.device, dtype=self.integration_dtype)

    def _lerp(self, guide, noise, keep: float, add: float):
        return (keep * guide.to(self.integration_dtype)
                + add * noise.to(self.integration_dtype))

    def to_pixels(self, state, differentiable: bool = False):
        """Native pixel BCHW -> canonical BHWC.  A transpose, differentiable when asked."""
        if differentiable:
            return state.permute(0, 2, 3, 1)                 # BCHW -> BHWC, graph intact
        arr = state.detach().float().clamp(-1.0, 1.0).cpu().numpy().transpose(0, 2, 3, 1)
        return np.ascontiguousarray(arr.astype(np.float32))

    # ------------------------------------------------------------------ standard-flow dynamics
    def _forward_native(self, state, t_batch, labels):
        """Cast only the network inputs; promote the prediction straight back to FP32."""
        model_state = state.to(device=self.device, dtype=self.dtype)
        model_t = t_batch.to(device=self.device, dtype=self.dtype)
        if self.backend_kind == "hf_mirror":
            output = self._model(model_state, timestep=model_t, class_labels=labels,
                                 interpolate_pos_encoding=True)
            native = output.sample if hasattr(output, "sample") else output
        else:
            native = self._model(model_state, model_t, labels)
        self.count_forwards(1)
        return native.float()

    def velocity(self, state, s: float, conditioning: Conditioning):
        """v_theta(x, s), derived from JiT's clean-image prediction."""
        torch = self.torch
        t = native_time(s, self.spec.native_time_mapping)               # = s
        state = state.to(device=self.device, dtype=self.integration_dtype)
        n = len(state)
        labels = torch.as_tensor(np.asarray(conditioning.labels, np.int64),
                                 device=self.device, dtype=torch.long)
        t_batch = torch.full((n,), float(t), device=self.device, dtype=self.integration_dtype)

        g = {**self.spec.guidance, **(conditioning.guidance or {})}
        scale = g.get("scale")
        interval = g.get("interval") or (0.0, 1.0)
        t_eps = float(g.get("t_eps", self.t_eps))

        conditional = self._forward_native(state, t_batch, labels)
        clean = conditional

        if scale is not None and float(scale) != 1.0:
            null_labels = torch.full_like(labels, self.spec.null_label)
            unconditional = self._forward_native(state, t_batch, null_labels)
            low, high = float(interval[0]), float(interval[1])
            # Guidance is gated per sample by the native-time interval, as in the source.
            active = (t_batch < high) & ((t_batch > low) if low != 0.0
                                         else torch.ones_like(t_batch, dtype=torch.bool))
            effective = torch.where(active, torch.full_like(t_batch, float(scale)),
                                    torch.ones_like(t_batch))
            sv = effective.reshape(-1, *([1] * (state.ndim - 1)))
            clean = unconditional + sv * (conditional - unconditional)

        t_view = t_batch.reshape(-1, *([1] * (state.ndim - 1)))
        denominator = (1.0 - t_view).clamp_min(t_eps)
        return (clean - state) / denominator

    def release(self) -> None:
        self._model = None
        super().release()

    # ------------------------------------------------------------------ sanity checks
    def sanity_checks(self) -> Dict[str, Any]:
        return {"identity_pixels": self._check_identity,
                "dtype_policy": self._check_dtype_policy,
                "clean_to_velocity": self._check_clean_to_velocity,
                "t_eps": self._check_t_eps,
                "guidance": self._check_guidance}

    def _check_identity(self, ctx) -> Tuple[bool, str]:
        native = self.encode_pixels(ctx["pixels"])
        back = self.to_pixels(native)
        ok = (tuple(native.shape[1:]) == tuple(self.spec.native_shape)
              and np.allclose(back, ctx["pixels"], atol=1e-5))
        return ok, "BHWC<->BCHW round-trip exact; native %s" % (tuple(native.shape),)

    def _check_dtype_policy(self, ctx) -> Tuple[bool, str]:
        state = self.prior_sample(range(2))
        ok = (state.dtype == self.torch.float32 and self.dtype != self.torch.float16)
        return ok, ("MPC/integration dtype=%s, network dtype=%s (FP16 avoided)"
                    % (state.dtype, self.dtype))

    def _check_clean_to_velocity(self, ctx) -> Tuple[bool, str]:
        torch = self.torch
        with torch.no_grad():
            state = self.prior_sample(range(2))
            cond = Conditioning(labels=np.asarray(ctx["conditioning"].labels[:1],
                                                  np.int32).repeat(2))
            t = 0.4
            v = self.velocity(state, t, cond)
            recovered_clean = state + max(1.0 - t, self.t_eps) * v
        ok = bool(torch.isfinite(v).all()) and bool(recovered_clean.abs().max() < 50)
        return ok, ("native=clean; derived v recovers x1_hat with |x1_hat|max=%.3f"
                    % float(recovered_clean.abs().max()))

    def _check_t_eps(self, ctx) -> Tuple[bool, str]:
        torch = self.torch
        with torch.no_grad():
            state = self.prior_sample(range(2))
            cond = Conditioning(labels=np.asarray(ctx["conditioning"].labels[:1],
                                                  np.int32).repeat(2))
            v = self.velocity(state, 0.999, cond)
        return bool(torch.isfinite(v).all()), \
            "s=0.999 -> finite velocity, |v|max=%.3f (t_eps=%.3f)" % (float(v.abs().max()),
                                                                      self.t_eps)

    def _check_guidance(self, ctx) -> Tuple[bool, str]:
        torch = self.torch
        with torch.no_grad():
            state = self.prior_sample(range(2))
            cond = Conditioning(labels=np.asarray(ctx["conditioning"].labels[:1],
                                                  np.int32).repeat(2))
            lo, hi = self.spec.guidance.get("interval") or (0.0, 1.0)
            self.reset_counters()
            self.velocity(state, min(0.5, (lo + hi) / 2), cond)
            forwards = self.forward_counter
        return forwards == 2, ("inside the guidance interval [%.2f, %.2f]: %d network forwards "
                               "(conditional + unconditional)" % (lo, hi, forwards))


@register_adapter("jit")
def _make_jit(registry: Dict[str, Any], context: Dict[str, Any]) -> JiTAdapter:
    return JiTAdapter(registry, context["repo_paths"]["jit"], context["ckpt_cache"],
                      context["torch_device"], context["torch_dtypes"]["jit"])
