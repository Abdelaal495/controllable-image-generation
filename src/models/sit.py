"""SiT adapter -- latent interpolant model with a Stable Diffusion VAE (BCHW 4x32x32).

Preserved from the notebooks and fully usable, but NOT enabled in the default configuration
(see configs/experiments.yaml for how to switch it on).

Loading, the official CFG rule (guidance applied to the first three latent channels only),
the SD-VAE `scaling_factor` and the mixed-precision timestep patch are unchanged.

TWO POLICIES, documented rather than hidden:
  * the MPC state and control are FP32; only the NETWORK runs in the checkpoint's dtype.
    Adam on a bf16 control would be numerically pointless.
  * `velocity` runs under the caller's autograd mode (no torch.inference_mode), so RHC with
    K>1 and MPC-delta_t can differentiate through v_theta.  SDEdit wraps its own loop in
    torch.no_grad() instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..utils import (STANDARD_FLOW, assert_pixel_batch, gaussian_noise, load_python_module,
                     native_time, pixel_fingerprint, pushd, record_time, timed)
from .base import (AdapterSpec, Conditioning, StandardFlowAdapter, register_adapter,
                   register_prefetch)


class SiTAdapter(StandardFlowAdapter):

    def __init__(self, registry: Dict[str, Any], repo_dir: Path, device, dtype):
        import torch
        super().__init__("sit", registry)
        self.torch = torch
        self.repo_dir = Path(repo_dir).resolve()   # pushd() chdirs here
        self.device = device
        self.dtype = dtype                      # network compute dtype
        self.integration_dtype = torch.float32  # MPC / solver state dtype
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        import time
        import types
        torch = self.torch
        t_start = time.perf_counter()
        cfg = self.registry
        variant = cfg["variant"]
        sit_models = load_python_module("sit_official_models", self.repo_dir / "models.py")
        sit_download = load_python_module("sit_official_download", self.repo_dir / "download.py")
        if variant not in sit_models.SiT_models:
            raise KeyError("Unknown SiT architecture: %r" % variant)

        model = sit_models.SiT_models[variant](input_size=32)
        if cfg["checkpoint"] == "official":
            with pushd(self.repo_dir):
                state_dict = sit_download.find_model("SiT-XL-2-256x256.pt")
            checkpoint_source = "official_download_via_repository"
        else:
            path = Path(cfg["checkpoint"])
            if not path.exists():
                raise FileNotFoundError(path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = loaded["ema"] if isinstance(loaded, dict) and "ema" in loaded else loaded
            checkpoint_source = str(path)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError("SiT checkpoint mismatch. missing=%s unexpected=%s"
                               % (missing, unexpected))
        model.eval().requires_grad_(False)      # no parameter gradients are ever allocated
        model.to(device=self.device, dtype=self.dtype)

        # Mixed-precision patch: the official sinusoidal timestep embedding is computed in
        # FP32 even when the MLP weights are reduced precision.  Cast before the MLP.
        def _mixed_precision_safe_timestep_forward(self_, t):
            t_freq = self_.timestep_embedding(t, self_.frequency_embedding_size)
            w = self_.mlp[0].weight
            return self_.mlp(t_freq.to(device=w.device, dtype=w.dtype))

        model.t_embedder.forward = types.MethodType(_mixed_precision_safe_timestep_forward,
                                                    model.t_embedder)
        self._model = model

        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(cfg["vae_id"], torch_dtype=self.dtype)
        vae.eval().requires_grad_(False)
        vae.to(self.device)
        self._vae = vae
        self.scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

        load_s = time.perf_counter() - t_start
        record_time("load/sit", load_s)
        n_params = sum(p.numel() for p in model.parameters())

        self.spec = AdapterSpec(
            name="sit", display_name=cfg["display_name"], dynamics_family=STANDARD_FLOW,
            framework="torch", state_space="latent", native_shape=tuple(cfg["native_shape"]),
            layout="BCHW", pixel_resolution=cfg["pixel_resolution"],
            prediction_kind="velocity", native_time_mapping=cfg["native_time_mapping"],
            batch_size=int(cfg["batch_size"]), fixed_batch_shape=False,
            num_classes=int(cfg["num_classes"]), null_label=int(cfg["null_label"]),
            guidance=dict(cfg["guidance"]),
            euler_final_step_for_heun=bool(cfg.get("euler_final_step_for_heun", False)),
            checkpoint={"variant": variant, "checkpoint_source": checkpoint_source,
                        "code_repository": cfg["repo_url"], "vae_id": cfg["vae_id"],
                        "vae_scaling_factor": self.scaling_factor,
                        "vae_encode_mode": cfg["vae_encode_mode"],
                        "prediction_type": "velocity", "path_type": "Linear",
                        "network_dtype": str(self.dtype), "mpc_state_dtype": "float32",
                        "parameters": int(n_params), "load_seconds": round(load_s, 2)})

    @property
    def vae_dtype(self):
        return next(self._vae.parameters()).dtype

    # ------------------------------------------------------------------ canonical interface
    def to_native_noise(self, noise: np.ndarray):
        return self.torch.from_numpy(np.ascontiguousarray(noise)).to(
            device=self.device, dtype=self.integration_dtype)

    def encode_pixels(self, pixels: np.ndarray):
        """Canonical (N,256,256,3) [-1,1] BHWC -> SCALED SiT latent (N,4,32,32) BCHW.

        The SD-VAE `scaling_factor` convention is SiT's own; the state lives in the scaled
        latent space, which is why the guide must be scaled the same way.
        """
        torch = self.torch
        assert_pixel_batch(pixels, "sit guide", self.spec.pixel_resolution)
        with torch.no_grad(), timed("encode/sit"):
            x = torch.from_numpy(np.ascontiguousarray(
                np.asarray(pixels, np.float32).transpose(0, 3, 1, 2)))   # BHWC -> BCHW
            x = x.to(device=self.device, dtype=self.vae_dtype)           # already in [-1,1]
            posterior = self._vae.encode(x).latent_dist
            mode = self.registry["vae_encode_mode"]
            if mode == "mean":
                unscaled = posterior.mean
            elif mode == "sample":
                eps = torch.from_numpy(gaussian_noise(
                    tuple(posterior.mean.shape), "sit", "vae_encode",
                    pixel_fingerprint(pixels))).to(device=self.device,
                                                   dtype=posterior.mean.dtype)
                unscaled = posterior.mean + posterior.std * eps
            else:
                raise ValueError("sit.vae_encode_mode must be 'mean' or 'sample'")
            return (unscaled * self.scaling_factor).to(self.integration_dtype)

    def _lerp(self, guide, noise, keep: float, add: float):
        return (keep * guide.to(self.integration_dtype)
                + add * noise.to(self.integration_dtype))

    def to_pixels(self, state, differentiable: bool = False):
        """Scaled SiT latent (BCHW) -> canonical (N,256,256,3) BHWC.

        differentiable=True keeps the SD-VAE decoder in the graph, which is what makes a
        pixel-space measurement loss usable as a terminal cost for a latent flow model.
        """
        torch = self.torch
        if differentiable:
            latents = state.to(dtype=self.vae_dtype) / self.scaling_factor
            decoded = self._vae.decode(latents).sample                   # BCHW, ~[-1,1]
            return decoded.to(self.integration_dtype).permute(0, 2, 3, 1)   # -> BHWC
        with torch.no_grad(), timed("decode/sit"):
            latents = state.detach().to(device=self.device,
                                        dtype=self.vae_dtype) / self.scaling_factor
            out = []
            bs = max(1, self.spec.batch_size)
            for i in range(0, len(latents), bs):
                decoded = self._vae.decode(latents[i:i + bs]).sample
                out.append(decoded.float().clamp(-1.0, 1.0).cpu().numpy())
            pixels = np.concatenate(out, axis=0).transpose(0, 2, 3, 1)   # BCHW -> BHWC
        return np.ascontiguousarray(pixels.astype(np.float32))

    # ------------------------------------------------------------------ standard-flow dynamics
    def velocity(self, state, s: float, conditioning: Conditioning):
        """v_theta(x, s).  SiT predicts velocity natively, so nothing is derived here.

        Official SiT classifier-free guidance is preserved verbatim: the guided velocity
        replaces only the first three latent channels.
        """
        torch = self.torch
        t = native_time(s, self.spec.native_time_mapping)               # = s
        state = state.to(device=self.device, dtype=self.integration_dtype)
        n = len(state)
        labels = torch.as_tensor(np.asarray(conditioning.labels, np.int64),
                                 device=self.device, dtype=torch.long)
        t_batch = torch.full((n,), float(t), device=self.device, dtype=self.dtype)

        g = {**self.spec.guidance, **(conditioning.guidance or {})}
        scale = g.get("scale")
        mode = g.get("mode") or "official_first3"

        model_state = state.to(self.dtype)
        conditional = self._model(model_state, t_batch, labels)
        self.count_forwards(1)
        native = conditional

        if scale is not None and float(scale) != 1.0:
            null_labels = torch.full_like(labels, self.spec.null_label)
            unconditional = self._model(model_state, t_batch, null_labels)
            self.count_forwards(1)
            full = unconditional + float(scale) * (conditional - unconditional)
            if mode == "official_first3":
                native = torch.cat([full[:, :3], conditional[:, 3:]], dim=1)
            elif mode == "all_channels":
                native = full
            else:
                raise ValueError("Unknown SiT CFG mode: %r" % mode)
        return native.to(self.integration_dtype)

    def release(self) -> None:
        self._model = None
        self._vae = None
        super().release()

    # ------------------------------------------------------------------ sanity checks
    def sanity_checks(self) -> Dict[str, Any]:
        return {"vae_roundtrip": self._check_encode_decode,
                "time_convention": self._check_time_convention,
                "velocity_semantics": self._check_velocity}

    def _check_encode_decode(self, ctx) -> Tuple[bool, str]:
        from ..metrics import psnr
        orig = ctx["pixels"]
        latent = self.encode_pixels(orig)
        recon = self.to_pixels(latent)
        ctx["reconstruction"] = recon
        mae = float(np.mean(np.abs(recon - orig)))
        shape_ok = tuple(latent.shape[1:]) == tuple(self.spec.native_shape)
        return (shape_ok and bool(self.torch.isfinite(latent).all()) and mae < 0.15
                and psnr(recon, orig) > 18.0), \
            "latent %s scaled by %.5f | MAE=%.4f PSNR=%.2f dB" % (
                tuple(latent.shape), self.scaling_factor, mae, psnr(recon, orig))

    def _check_time_convention(self, ctx) -> Tuple[bool, str]:
        ok = (abs(native_time(0.0, self.spec.native_time_mapping) - 0.0) < 1e-12
              and abs(native_time(1.0, self.spec.native_time_mapping) - 1.0) < 1e-12)
        return ok, "canonical s maps to native t identically (s=0 noise, s=1 data)"

    def _check_velocity(self, ctx) -> Tuple[bool, str]:
        torch = self.torch
        with torch.no_grad():
            state = self.prior_sample(range(2))
            cond = Conditioning(labels=np.asarray(ctx["conditioning"].labels[:1],
                                                  np.int32).repeat(2))
            v = self.velocity(state, 0.4, cond)
            x1 = state + (1.0 - 0.4) * v
            pix = self.to_pixels(x1)
        ok = (bool(torch.isfinite(v).all()) and v.dtype == torch.float32
              and pix.shape[1:] == (self.spec.pixel_resolution, self.spec.pixel_resolution, 3))
        return ok, ("native prediction = velocity, |v|max=%.3f, one-step Euler endpoint "
                    "decodes to %s" % (float(v.abs().max()), pix.shape))


@register_adapter("sit")
def _make_sit(registry: Dict[str, Any], context: Dict[str, Any]) -> SiTAdapter:
    return SiTAdapter(registry, context["repo_paths"]["sit"], context["torch_device"],
                      context["torch_dtypes"]["sit"])


@register_prefetch("sit")
def _prefetch_sit(registry: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch the SiT checkpoint (through the repository's own downloader) and the SD-VAE."""
    info: Dict[str, Any] = {}
    repo_dir = Path(context["repo_paths"]["sit"]).resolve()
    if registry["checkpoint"] == "official":
        sit_download = load_python_module("sit_official_download", repo_dir / "download.py")
        with pushd(repo_dir):
            sit_download.find_model("SiT-XL-2-256x256.pt")
        info["checkpoint"] = "SiT-XL-2-256x256.pt (cached under %s)" % repo_dir
    from diffusers import AutoencoderKL
    AutoencoderKL.from_pretrained(registry["vae_id"])
    info["vae"] = registry["vae_id"]
    return info
