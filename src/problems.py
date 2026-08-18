"""The ONE inverse-problem layer.  Every task is built here exactly once.

    prepare one observation  ->  form one corrupted initial state  ->  apply one strategy

SDEdit, MPC-RHC and MPC-delta_t all consume the same `InverseProblem` object, so no
comparison in this repository can be confounded by a different noise draw, a different
mask or a different stroke geometry.

Six problems
------------
    denoising          A(x) = x                       y = x* + eta
    deblur             separable Gaussian blur        y = A(x*) + eta
    super_resolution   x[:, ::f, ::f, :]              y is genuinely low resolution
    box_inpaint        M (o) x, central box removed   noise applies only where observed
    random_inpaint     M (o) x, random pixels dropped  ditto
    stroke_painting    A_G(x), frozen SLIC geometry   y = A_G(x*), noiseless by default

Every operator is written ONCE against the three-primitive backend shim in `utils` and then
executed identically by NumPy (to build y), PyTorch (JiT/SiT objectives) and JAX (pMF/iMF
objectives).  Reflect padding is an index gather rather than each framework's own `pad`, so
the three backends agree bitwise rather than merely approximately.

Initialisation guides
---------------------
The guide g(y) exists ONLY to build the t0 < 1 starting state; the objective always uses the
measurement y itself.  Each problem defines exactly one guide:

    denoising / deblur / inpainting / stroke   g(y) = y
    super_resolution                           g(y) = BicubicUpsample(y) to 256x256

For the two inpainting tasks the guide is deliberately the zero-filled masked observation.
Telea / Navier-Stokes / RePaint-style prefill and hard per-step projection are NOT available:
they would change what the generative model is being asked to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import PROBLEM_DECLARATIONS, STROKE_PRESETS, ProblemRequest
from .utils import (Backend, NUMPY_BACKEND, assert_pixel_batch, canonical_params_key,
                    derive_rng, gaussian_kernel_1d, gaussian_noise, mask_parts,
                    measurement_noise_parts, pixel_fingerprint, separable_gaussian_blur,
                    stroke_geometry_parts, timed, to_float, to_uint8, CANONICAL_RESOLUTION,
                    SEED_RECIPES)


# =====================================================================================
# Stroke geometry -- extracted once with the ORIGINAL algorithm, then frozen
# =====================================================================================
@dataclass
class StrokeGeometry:
    """Frozen spatial geometry G for one image.

    Produced by the SDEdit notebook's SLIC stroke algorithm (superpixels, per-segment PCA,
    a line along the principal direction, positional jitter, drawing order).  Extraction is
    NOT differentiable and does not need to be: it happens once, outside the MPC graph.

    After freezing, `render_strokes` is a pure tensor function of the image.

        segment_map    (H,W) int32   which superpixel each pixel belongs to
        source_index   (H,W) int32   which superpixel's mean colour each pixel DISPLAYS
                                     (the stroke owner where a stroke covers the pixel,
                                      otherwise the pixel's own superpixel)
        counts         (S,)  float32 pixels per superpixel, for the differentiable mean
        num_segments   int
    """
    segment_map: np.ndarray
    source_index: np.ndarray
    counts: np.ndarray
    num_segments: int
    preset: str
    params: Dict[str, Any]

    def fingerprint(self) -> str:
        return pixel_fingerprint(np.concatenate(
            [self.segment_map.reshape(-1).astype(np.float32),
             self.source_index.reshape(-1).astype(np.float32)]))


def _encode_id(value: int) -> Tuple[int, int, int]:
    return ((value >> 16) & 255, (value >> 8) & 255, value & 255)


def extract_stroke_geometry(img_uint8: np.ndarray, n_segments: int = 200,
                            compactness: int = 10, stroke_width: int = 6,
                            rng_seed: int = 0) -> StrokeGeometry:
    """Run the original stroke algorithm and keep only its GEOMETRY.

    The control flow, the RandomState draw order, the 80th-percentile half-length, the
    clipping and the PIL line rasterisation are identical to
    `make_stroke_painting` in the SDEdit notebook, so the strokes land in exactly the same
    places.  What differs is that colours are not baked in here: instead of drawing the
    segment's mean colour, we draw the segment's INDEX, giving an owner map that the
    differentiable renderer can colour later from any image.
    """
    from PIL import ImageDraw
    from skimage.segmentation import slic

    H, W, _ = img_uint8.shape
    rng = np.random.RandomState(rng_seed)
    segments = slic(img_uint8, n_segments=n_segments, compactness=compactness,
                    start_label=0, channel_axis=-1)
    seg_ids = np.unique(segments)
    # Compact the labels to 0..S-1 so the renderer can index contiguously.
    remap = np.zeros(int(seg_ids.max()) + 1, np.int32)
    for new_id, old_id in enumerate(seg_ids):
        remap[old_id] = new_id
    segment_map = remap[segments].astype(np.int32)
    num_segments = int(len(seg_ids))
    counts = np.bincount(segment_map.reshape(-1),
                         minlength=num_segments).astype(np.float32)

    # Owner canvas: colour 0 means "no stroke here"; a stroke writes (index + 1).
    owner_img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(owner_img)
    for new_id, seg_id in enumerate(seg_ids):
        mask = segments == seg_id
        ys, xs = np.where(mask)
        if len(ys) < 3:                       # the original skips BEFORE drawing from rng
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        coords = np.stack([xs - cx, ys - cy], axis=1).astype(np.float32)
        cov = (coords.T @ coords) / len(coords)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, int(np.argmax(eigvals))]
        half_len = float(np.percentile(np.abs(coords @ major), 80))
        half_len = float(np.clip(half_len, 3, min(H, W) // 4))
        cx += rng.uniform(-half_len * 0.1, half_len * 0.1)
        cy += rng.uniform(-half_len * 0.1, half_len * 0.1)
        dx, dy = major[0] * half_len, major[1] * half_len
        draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)],
                  fill=_encode_id(new_id + 1), width=stroke_width)

    owner = np.array(owner_img).astype(np.int32)
    owner_id = (owner[..., 0] << 16) | (owner[..., 1] << 8) | owner[..., 2]
    source_index = np.where(owner_id > 0, owner_id - 1, segment_map).astype(np.int32)
    return StrokeGeometry(segment_map=segment_map, source_index=source_index, counts=counts,
                          num_segments=num_segments, preset="",
                          params={"n_segments": int(n_segments), "compactness": int(compactness),
                                  "stroke_width": int(stroke_width), "rng_seed": int(rng_seed)})


def make_stroke_painting_reference(img_uint8: np.ndarray, n_segments: int = 200,
                                   compactness: int = 10, stroke_width: int = 6,
                                   rng_seed: int = 0) -> np.ndarray:
    """The ORIGINAL SDEdit stroke transform, verbatim.

    Kept for the parity test only.  It is never the benchmark forward operator, because it
    is not differentiable: the measurement is produced by the frozen-geometry renderer so
    that y and the MPC objective use the same mathematical operator.
    """
    from PIL import ImageDraw, ImageFilter
    from skimage.segmentation import slic
    H, W, _ = img_uint8.shape
    rng = np.random.RandomState(rng_seed)
    segments = slic(img_uint8, n_segments=n_segments, compactness=compactness,
                    start_label=0, channel_axis=-1)
    canvas_arr = np.zeros_like(img_uint8)
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        canvas_arr[mask] = img_uint8[mask].mean(axis=0).astype(np.uint8)
    canvas = Image.fromarray(canvas_arr)
    draw = ImageDraw.Draw(canvas)
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        avg_color = tuple(int(c) for c in img_uint8[mask].mean(axis=0))
        ys, xs = np.where(mask)
        if len(ys) < 3:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        coords = np.stack([xs - cx, ys - cy], axis=1).astype(np.float32)
        cov = (coords.T @ coords) / len(coords)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, int(np.argmax(eigvals))]
        half_len = float(np.percentile(np.abs(coords @ major), 80))
        half_len = float(np.clip(half_len, 3, min(H, W) // 4))
        cx += rng.uniform(-half_len * 0.1, half_len * 0.1)
        cy += rng.uniform(-half_len * 0.1, half_len * 0.1)
        dx, dy = major[0] * half_len, major[1] * half_len
        draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=avg_color, width=stroke_width)
    return np.array(canvas.filter(ImageFilter.GaussianBlur(radius=0.8)))


# =====================================================================================
# The problem object
# =====================================================================================
@dataclass
class InverseProblem:
    """One (problem, parameters, image set) instance.

    Built ONCE per ProblemRequest and reused by every model, method and sweep value.
    """
    name: str
    key: str
    sigma: float
    params: Dict[str, Any]
    ground_truth: np.ndarray                       # (N,256,256,3) x*
    measurement: np.ndarray                        # y, canonical float32
    display_measurement: np.ndarray                # (N,256,256,3) for figures only
    initialization_guide: np.ndarray               # g(y), (N,256,256,3)
    guide_mode: str
    mask: Optional[np.ndarray] = None              # (N,256,256,1) float32, 1 = observed
    geometry: Optional[List[StrokeGeometry]] = None
    image_ids: Tuple[str, ...] = ()
    labels: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _cache: Dict[Any, Any] = field(default_factory=dict, repr=False)

    # -- backend-cached constants ---------------------------------------------------
    def _c(self, B: Backend, name: str, arr):
        key = (B.key, name)
        if key not in self._cache:
            self._cache[key] = B.const(arr)
        return self._cache[key]

    def _i(self, B: Backend, name: str, arr):
        key = (B.key, "idx:" + name)
        if key not in self._cache:
            self._cache[key] = B.index(arr)
        return self._cache[key]

    # -- stroke helpers --------------------------------------------------------------
    def _stroke_tables(self):
        """Flatten per-image geometry into one global segment indexing.

        Segment j of image i becomes global segment i*S_max + j, so the whole batch is
        rendered with a single segment-sum and a single gather in every backend.
        """
        if "stroke_tables" not in self._cache:
            geoms = self.geometry or []
            n = len(geoms)
            s_max = max(int(g.num_segments) for g in geoms)
            h, w = geoms[0].segment_map.shape
            seg = np.stack([g.segment_map for g in geoms]).astype(np.int64)
            src = np.stack([g.source_index for g in geoms]).astype(np.int64)
            offsets = (np.arange(n, dtype=np.int64) * s_max).reshape(n, 1, 1)
            counts = np.zeros((n * s_max,), np.float32)
            for i, g in enumerate(geoms):
                counts[i * s_max:i * s_max + int(g.num_segments)] = g.counts
            self._cache["stroke_tables"] = {
                "seg_flat": (seg + offsets).reshape(-1),
                "src_flat": (src + offsets).reshape(-1),
                "inv_counts": (1.0 / np.maximum(counts, 1.0)).astype(np.float32),
                "num_global": int(n * s_max), "shape": (n, h, w),
            }
        return self._cache["stroke_tables"]

    def _render_strokes(self, x, B: Backend):
        """Differentiable stroke renderer A_G(x).  No PIL, NumPy or skimage on this path.

            c_j(x) = sum_p M_j(p) x(p) / sum_p M_j(p)      (segment mean, differentiable)
            canvas(p) = c_{source_index(p)}(x)             (base colours + stroke overwrite,
                                                            preserving the drawing order)
            A_G(x) = GaussianBlur_sigma(canvas)
        """
        tables = self._stroke_tables()
        n, h, w = tables["shape"]
        c = int(x.shape[-1])
        seg = self._i(B, "stroke_seg", tables["seg_flat"])
        src = self._i(B, "stroke_src", tables["src_flat"])
        inv_counts = self._c(B, "stroke_inv_counts", tables["inv_counts"].reshape(1, -1, 1))

        flat = x.reshape(1, n * h * w, c)
        sums = B.segment_sum(flat, seg, tables["num_global"])          # (1, n*S, C)
        colors = sums * inv_counts
        canvas = B.take(colors, src, 1).reshape(n, h, w, c)
        kernel = gaussian_kernel_1d(float(self.params.get("blur_sigma", 0.8)),
                                    int(self.params.get("blur_kernel_size", 7)))
        return separable_gaussian_blur(canvas, kernel, B, self._cache, tag="stroke_blur")

    # -- the forward operator --------------------------------------------------------
    def apply(self, x, B: Backend = NUMPY_BACKEND):
        """A(x) for canonical BHWC pixels `x` living in backend `B`.  Differentiable."""
        if self.name == "denoising":
            return x
        if self.name == "super_resolution":
            f = int(self.params["factor"])
            return x[:, ::f, ::f, :]
        if self.name in ("random_inpaint", "box_inpaint"):
            return x * self._c(B, "mask", self.mask)
        if self.name == "deblur":
            kernel = gaussian_kernel_1d(float(self.params["blur_sigma"]),
                                        int(self.params["kernel_size"]))
            return separable_gaussian_blur(x, kernel, B, self._cache, tag="deblur")
        if self.name == "stroke_painting":
            return self._render_strokes(x, B)
        raise ValueError("Unknown inverse problem %r" % self.name)

    def measurement_tensor(self, B: Backend):
        """y in backend `B`, cached so it is uploaded to the device only once."""
        return self._c(B, "y", self.measurement)

    # -- bookkeeping ------------------------------------------------------------------
    def subset(self, indices: Sequence[int]) -> "InverseProblem":
        """A view over a subset / repetition of the images (batch chunks and padding)."""
        idx = np.asarray(list(indices), np.int64)
        return InverseProblem(
            name=self.name, key=self.key, sigma=self.sigma, params=dict(self.params),
            ground_truth=self.ground_truth[idx], measurement=self.measurement[idx],
            display_measurement=self.display_measurement[idx],
            initialization_guide=self.initialization_guide[idx], guide_mode=self.guide_mode,
            mask=None if self.mask is None else self.mask[idx],
            geometry=None if self.geometry is None else [self.geometry[i] for i in idx],
            image_ids=tuple(self.image_ids[i] for i in idx),
            labels=None if self.labels is None else self.labels[idx],
            metadata=dict(self.metadata))

    def describe(self) -> str:
        extra = {k: v for k, v in self.params.items() if k not in ("factor",)}
        return "%-17s sigma=%.3f  y%s  guide=%-17s %s" % (
            self.name, self.sigma, tuple(self.measurement.shape), self.guide_mode, extra)

    def observed_fraction(self) -> float:
        total = float(np.prod(self.ground_truth.shape[1:]))
        if self.mask is not None:
            return float(self.mask[0].mean())
        return float(np.prod(self.measurement.shape[1:])) / total

    def to_metadata(self) -> Dict[str, Any]:
        meta = {"problem": self.name, "problem_key": self.key, "sigma": self.sigma,
                "params": dict(self.params), "guide_mode": self.guide_mode,
                "measurement_shape": list(self.measurement.shape),
                "observed_fraction": self.observed_fraction(),
                "image_ids": list(self.image_ids),
                "measurement_fingerprint": pixel_fingerprint(self.measurement),
                "guide_fingerprint": pixel_fingerprint(self.initialization_guide)}
        if self.mask is not None:
            meta["mask_fingerprint"] = pixel_fingerprint(self.mask)
        if self.geometry is not None:
            meta["geometry_fingerprints"] = [g.fingerprint() for g in self.geometry]
            meta["geometry_note"] = (
                "The stroke geometry G is DERIVED FROM THE SOURCE IMAGE and then frozen. "
                "Unlike a random inpainting mask, which is independent of image content, "
                "G encodes the image's own superpixel structure, so the stroke measurement "
                "carries more information about x* than its visual sparsity suggests.")
        meta.update(self.metadata)
        return meta


# =====================================================================================
# Construction -- deterministic, once per ProblemRequest
# =====================================================================================
def build_problem(request: ProblemRequest, ground_truth: np.ndarray,
                  image_ids: Sequence[str], labels: Optional[np.ndarray] = None,
                  verbose: bool = True) -> InverseProblem:
    """Build y = A(x*) + eta deterministically for one ProblemRequest.

    Measurement, mask and stroke-geometry seeds depend on (problem, canonical problem
    parameters, image id) only -- never on the experiment name, the model, the method or the
    order things run in.  Two differently-named experiments asking for the same
    specification provably share one y; changing any parameter provably creates a new one.
    """
    name = request.problem
    cfg = dict(PROBLEM_DECLARATIONS[name].default_params)
    cfg.update(request.params)
    assert_pixel_batch(ground_truth, "ground truth for %s" % name, CANONICAL_RESOLUTION)
    n = len(ground_truth)
    sigma = float(cfg.get("sigma", 0.0))
    params = {k: v for k, v in cfg.items() if k != "sigma"}
    params_key = canonical_params_key(cfg)
    mask = None
    geometry = None

    # ---------------------------------------------------------------- masks
    if name in ("random_inpaint", "box_inpaint"):
        h = w = CANONICAL_RESOLUTION
        masks = []
        for i in range(n):
            if name == "random_inpaint":
                rng = derive_rng(*mask_parts(name, params_key, image_ids[i]))
                keep = rng.random((h, w)) >= float(params["missing_fraction"])
                m = keep.astype(np.float32)[..., None]      # per-pixel, shared by channels
            else:
                box = int(params["box"])
                m = np.ones((h, w, 1), np.float32)
                y0, x0 = (h - box) // 2, (w - box) // 2
                m[y0:y0 + box, x0:x0 + box, :] = 0.0        # identical for every image
            masks.append(m)
        mask = np.stack(masks, axis=0)

    # ---------------------------------------------------------------- stroke geometry
    if name == "stroke_painting":
        preset_name = str(params.get("preset", "medium"))
        preset = dict(STROKE_PRESETS[preset_name])
        params.setdefault("blur_sigma", 0.8)
        params["blur_kernel_size"] = int(2 * int(np.ceil(3.0 * float(params["blur_sigma"]))) + 1)
        params.update(preset)
        geometry = []
        gt_u8 = to_uint8(ground_truth)
        for i in range(n):
            seed = int(derive_rng(*stroke_geometry_parts(params_key,
                                                         image_ids[i])).integers(0, 2 ** 31 - 1))
            with timed("geometry/stroke"):
                g = extract_stroke_geometry(gt_u8[i], rng_seed=seed, **preset)
            g.preset = preset_name
            geometry.append(g)
        if verbose:
            print("   stroke geometry frozen for %d image(s): preset=%s, segments=%s"
                  % (n, preset_name, [g.num_segments for g in geometry]))

    probe = InverseProblem(
        name=name, key=request.key, sigma=sigma, params=params,
        ground_truth=np.asarray(ground_truth, np.float32),
        measurement=np.zeros((n, 1, 1, 3), np.float32),
        display_measurement=np.zeros_like(ground_truth),
        initialization_guide=np.zeros_like(ground_truth),
        guide_mode=request.guide_mode, mask=mask, geometry=geometry,
        image_ids=tuple(image_ids),
        labels=None if labels is None else np.asarray(labels, np.int32))

    clean = np.asarray(probe.apply(ground_truth, NUMPY_BACKEND), np.float32)

    # Noise is drawn in MEASUREMENT space, once per (problem, params, image).
    if sigma > 0.0:
        noise = np.stack([gaussian_noise(clean.shape[1:],
                                         *measurement_noise_parts(name, params_key,
                                                                  image_ids[i]))
                          for i in range(n)], axis=0)
        if mask is not None:
            # Unobserved entries must not become observations: the residual A(x) - y is then
            # identically zero off the mask, which is the null-space situation MPC-Flow
            # analyses for K = 1.
            noise = noise * mask
        y = clean + sigma * noise
    else:
        y = clean.copy()

    probe.measurement = y.astype(np.float32)
    probe.display_measurement = _display_measurement(name, y, params)
    probe.initialization_guide = build_initialization_guide(probe, request.guide_mode)
    probe.metadata = {
        "measurement_seed_recipe": SEED_RECIPES["measurement"],
        "mask_seed_recipe": SEED_RECIPES["mask"] if mask is not None else None,
        "stroke_geometry_seed_recipe": (SEED_RECIPES["stroke_geometry"]
                                        if geometry is not None else None),
        "problem_params_key": params_key,
        "noise_in_measurement_space": True,
        "noise_masked": mask is not None,
        "classical_prefill": False,
    }
    return probe


def _display_measurement(name: str, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Figures only.  Never the objective and never the guide."""
    if name == "super_resolution":
        f = int(params["factor"])
        display = np.repeat(np.repeat(y, f, axis=1), f, axis=2)     # nearest-neighbour
    else:
        display = y.copy()
    return np.clip(display, -1.0, 1.0).astype(np.float32)


def build_initialization_guide(problem: InverseProblem, mode: str) -> np.ndarray:
    """g(y) -> canonical (N,256,256,3) float32 [-1,1].

    The measurement y is the ONLY input.  The ground truth is never consulted: that would
    leak the answer into the initialisation.
    """
    y = np.asarray(problem.measurement, np.float32)
    res = CANONICAL_RESOLUTION

    if mode in ("identity", "observed", "zero_fill"):
        # denoising: g(y) = y.  deblur / stroke: the observed image itself.
        # inpainting: the masked observation, missing entries exactly 0 (mid-grey) -- the
        # honest "no information" baseline.  NO classical prefill; see the module docstring.
        guide = y.copy()
    elif mode == "upsample_bicubic":
        # super-resolution: lift the low-resolution measurement into canonical pixel space.
        # This is an INITIALISATION choice; it is never the measurement operator A.
        out = []
        for i in range(len(y)):
            img = Image.fromarray(to_uint8(y[i]))
            out.append(to_float(np.array(img.resize((res, res), resample=Image.BICUBIC))))
        guide = np.stack(out, axis=0)
    else:                                                           # pragma: no cover
        raise ValueError("Unimplemented guide mode %r" % mode)

    guide = np.clip(np.asarray(guide, np.float32), -1.0, 1.0)
    assert_pixel_batch(guide, "initialisation guide (%s/%s)" % (problem.name, mode), res)
    return guide


class ProblemStore:
    """Realises every ProblemRequest exactly once and hands the instances out by key."""

    def __init__(self):
        self._by_key: Dict[str, InverseProblem] = {}

    def build_all(self, requests: Sequence[ProblemRequest], data_manager,
                  verbose: bool = True) -> Dict[str, InverseProblem]:
        for req in requests:
            if req.key in self._by_key:
                continue
            examples = data_manager.examples(req.num_images)
            with timed("problem/build/%s" % req.problem):
                self._by_key[req.key] = build_problem(
                    req, examples.images, examples.image_ids, examples.labels, verbose)
            if verbose:
                print("   %s" % self._by_key[req.key].describe())
        return self._by_key

    def get(self, key: str) -> InverseProblem:
        if key not in self._by_key:
            raise KeyError("Problem %r has not been built." % key)
        return self._by_key[key]

    def items(self):
        return self._by_key.items()

    def __len__(self) -> int:
        return len(self._by_key)


# =====================================================================================
# Terminal objective and control cost  (MPC-Flow section 16)
# =====================================================================================
VALID_PHI_NORMALIZATIONS = ("half_sum_squared", "sum_squared", "mean_squared",
                            "gaussian_likelihood", "half_mean_squared_per_measurement")
PER_MEASUREMENT_NORMALIZATION = "half_mean_squared_per_measurement"


def measurement_counts(problem: InverseProblem) -> np.ndarray:
    """m_b: the number of ACTUAL observed scalar measurements for each sample b.

    Not blindly the size of A(x): what matters is how many scalar numbers the residual
    really carries information about.

        denoising / deblur          H * W * C
        super_resolution            H_low * W_low * C   -- the real low-resolution y, NOT
                                                           the 256x256 bicubic guide
        box / random inpainting     C * (observed pixels)  -- masked entries contribute an
                                                           identically-zero residual and
                                                           must not dilute the mean
        stroke_painting             the scalar entries of the differentiable rendered
                                    measurement A_G(x); no "effective degrees of freedom"
                                    correction is invented for the stroke geometry

    Returned per sample because an inpainting mask is drawn per image, so m_b genuinely
    differs across a batch.
    """
    n = int(problem.measurement.shape[0])
    per_sample_entries = float(np.prod(problem.measurement.shape[1:]))
    if problem.mask is None:
        return np.full((n,), per_sample_entries, np.float32)
    # The mask is (N,H,W,1) and broadcasts over the C channels of A(x) = mask * x.
    mask = np.asarray(problem.mask, np.float32)
    channels = int(problem.measurement.shape[-1])
    per_image = mask.reshape(n, -1).sum(axis=1) * (channels / float(mask.shape[-1]))
    counts = np.maximum(per_image, 1.0).astype(np.float32)
    return counts


def _per_measurement_weights(problem: InverseProblem) -> np.ndarray:
    """1/m_b shaped (N,1,1,1) so it broadcasts against any BHWC residual."""
    counts = measurement_counts(problem)
    return (1.0 / counts).astype(np.float32).reshape(-1, 1, 1, 1)


def phi_log_scale(problem: InverseProblem, normalization: str) -> float:
    """Factor turning the OPTIMISED scalar into a per-image number for the loss history.

    The per-measurement objective is summed over the batch (see `make_phi`), which keeps
    each sample's gradient independent of the batch size.  A summed number is not
    comparable across batch sizes when a human reads it, so `loss_history` divides by the
    batch dimension.  Rows that were repeat-padded to fill a fixed compiled batch are
    duplicates of a real image, so the mean stays on a per-image scale.
    """
    if normalization != PER_MEASUREMENT_NORMALIZATION:
        return 1.0
    return 1.0 / max(1, int(problem.measurement.shape[0]))


def make_phi(problem: InverseProblem, B: Backend, normalization: str):
    """Build Phi(x_pixels) = data-fidelity(A(x), y) for one problem in one backend.

    `x_pixels` is canonical BHWC, and the whole path stays inside the caller's autograd
    graph: no NumPy conversion, no detach, no PIL.

    `half_mean_squared_per_measurement` (the PnP-Flow / D-Flow default) is

        F_b(x_b) = (1 / 2 m_b) * sum_{j in Omega_b} (A(x_b)_j - y_b,j)^2
        L_opt    = sum_b F_b                        <- what is optimised

    It is SUMMED, never averaged, across the batch: reconstruction variables are
    independent per image, so summing leaves each sample's gradient invariant to the batch
    size and to repeat padding.  Use `phi_log_scale` for the number you report.

    It deliberately carries NO 1/sigma^2 factor.  A sigma of 0.05 would otherwise inflate
    the objective 200-fold and silently move the useful gamma0 / lr by the same factor;
    those hyperparameters are swept instead.  `gaussian_likelihood` remains available as an
    explicit choice and keeps its 1/(2 sigma^2), but is rejected for a noiseless problem
    rather than falling back to a tiny epsilon.
    """
    if normalization not in VALID_PHI_NORMALIZATIONS:
        raise ValueError("Unknown phi_normalization %r" % normalization)
    if normalization == "gaussian_likelihood" and float(problem.sigma) <= 0.0:
        raise ValueError(
            "phi_normalization='gaussian_likelihood' is undefined for %s: sigma=%g. Its "
            "1/(2 sigma^2) factor would have to invent a noise level. Use "
            "'half_mean_squared_per_measurement' or 'half_sum_squared' instead."
            % (problem.name, float(problem.sigma)))
    y = problem.measurement_tensor(B)
    if normalization == "gaussian_likelihood":
        inv_two_sigma_sq = 1.0 / (2.0 * float(problem.sigma) ** 2)
    if normalization == PER_MEASUREMENT_NORMALIZATION:
        # One cached (N,1,1,1) constant per backend; no per-sample Python loop, and the
        # same expression in NumPy, Torch and JAX.
        half_inv_m = problem._c(B, "half_inv_measurements",
                                0.5 * _per_measurement_weights(problem))

    def phi(x_pixels):
        residual = problem.apply(x_pixels, B) - y
        sq = residual * residual
        if normalization == "half_sum_squared":
            return 0.5 * B.sum(sq)
        if normalization == "sum_squared":
            return B.sum(sq)
        if normalization == "mean_squared":
            return B.mean(sq)
        if normalization == PER_MEASUREMENT_NORMALIZATION:
            return B.sum(sq * half_inv_m)
        return inv_two_sigma_sq * B.sum(sq)          # gaussian_likelihood

    return phi


def make_control_cost(B: Backend, normalization: str):
    """||u||^2 with the configured normalisation."""
    if normalization not in ("sum_squared", "mean_squared"):
        raise ValueError("Unknown control_cost_normalization %r" % normalization)

    def cost(u):
        sq = u * u
        return B.sum(sq) if normalization == "sum_squared" else B.mean(sq)

    return cost
