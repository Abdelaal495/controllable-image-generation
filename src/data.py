"""Source images: one shared pool of real ImageNet-1k validation images.

Carried over from the MPC-Flow notebook (section 12).  The pool is sized by the largest
enabled experiment and loaded once; every experiment takes a PREFIX of it, so two
experiments requesting 4 and 8 images provably see the same first four.

True ImageNet labels are used for class conditioning (see README: the benchmark is
"reconstruction conditioned on the known source class").  Synthetic images are refused:
PSNR/SSIM/LPIPS against procedural textures say nothing about an ImageNet prior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .utils import assert_pixel_batch, get_hf_token, timed, to_float

# Verbatim from the source notebooks; the order is stable so image ids are meaningful seeds.
IMAGENET_EXAMPLES: List[Tuple[int, str]] = [
    (0, "tench"), (39, "iguana"), (81, "ptarmigan"), (88, "macaw"),
    (207, "golden_retriever"), (281, "tabby_cat"), (291, "lion"),
    (323, "monarch_butterfly"), (340, "zebra"), (417, "balloon"),
    (562, "fountain"), (698, "palace"), (717, "pickup_truck"), (817, "sports_car"),
    (825, "stone_wall"), (947, "mushroom"), (973, "coral_reef"), (979, "valley"),
    (985, "daisy"), (388, "giant_panda"), (292, "tiger"), (986, "sunflower"),
    (980, "volcano"), (663, "monastery"), (525, "dam"), (991, "coral_fungus"),
    (319, "dragonfly"), (697, "parachute"), (581, "geyser"), (84, "peacock"),
    (49, "crocodile"), (360, "otter"),
]


@dataclass
class SourceExamples:
    """The shared, model-independent ground-truth dataset."""
    images: np.ndarray                 # (N,256,256,3) float32 [-1,1]
    labels: np.ndarray                 # (N,) int32 true ImageNet class ids
    names: List[str]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.images)

    @property
    def image_ids(self) -> List[str]:
        """Stable per-image identity used by every seed recipe and every result record."""
        return ["%03d_%s" % (int(self.labels[i]), self.names[i]) for i in range(len(self))]

    def prefix(self, n: int) -> "SourceExamples":
        if n > len(self):
            raise ValueError("Requested %d images but the pool holds %d." % (n, len(self)))
        return SourceExamples(self.images[:n], self.labels[:n], list(self.names[:n]),
                              self.source, dict(self.metadata))


def center_crop(pil_image, size: int):
    """Resize the short side to `size`, then take the central square crop."""
    scale = size / min(pil_image.size)
    pil_image = pil_image.resize(tuple(max(size, round(x * scale)) for x in pil_image.size),
                                 resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy = (arr.shape[0] - size) // 2
    cx = (arr.shape[1] - size) // 2
    return Image.fromarray(arr[cy:cy + size, cx:cx + size])


class DataManager:
    """Owns the shared source pool.  Loads it once, at the size the plan requires."""

    def __init__(self, config: Dict[str, Any], cache_dir: Path):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = int(config["data"].get("image_size", 256))
        self.source_kind = config["data"].get("source", "hf_imagenet_val")
        self._pool: Optional[SourceExamples] = None

    def pool(self, n: int) -> SourceExamples:
        if self._pool is None or len(self._pool) < n:
            with timed("data/load_pool"):
                self._pool = self._load(max(n, len(self._pool) if self._pool else 0))
            assert_pixel_batch(self._pool.images, "source images", self.image_size)
            if not ((0 <= self._pool.labels) & (self._pool.labels < 1000)).all():
                raise RuntimeError("Source labels outside [0,1000).")
            print("Loaded %d source examples from %r  shape=%s  range=[%.3f, %.3f]"
                  % (len(self._pool), self._pool.source, self._pool.images.shape,
                     self._pool.images.min(), self._pool.images.max()))
        return self._pool

    def examples(self, n: int) -> SourceExamples:
        """The first n images of the shared pool -- a prefix, so subsets always agree."""
        return self.pool(n).prefix(n)

    # ------------------------------------------------------------------ loaders
    def _load(self, n: int) -> SourceExamples:
        if self.source_kind == "hf_imagenet_val":
            return self._load_imagenet(n)
        if self.source_kind == "local_folder":
            return self._load_local(n, self.config["data"]["local_folder"])
        raise ValueError("Unsupported data.source %r" % self.source_kind)

    def _load_imagenet(self, n: int) -> SourceExamples:
        cache = self.cache_dir / ("imagenet_val_%d_%d.npz" % (n, self.image_size))
        if cache.exists():
            with np.load(cache, allow_pickle=True) as z:
                return SourceExamples(z["images"], z["labels"], list(z["names"]),
                                      "hf_imagenet_val", {"cache": str(cache)})
        if n > len(IMAGENET_EXAMPLES):
            raise ValueError("The curated ImageNet list has %d entries but %d were requested. "
                             "Use data.source: local_folder for a larger pool."
                             % (len(IMAGENET_EXAMPLES), n))
        token = get_hf_token(required=True)
        try:
            from datasets import load_dataset
        except ImportError as exc:
            # `datasets` needs pyarrow, which on Alliance clusters comes from the `arrow`
            # module rather than pip.  The cache this builds is a single small .npz that is
            # fully portable, so it is often easier to build it elsewhere and copy it in
            # than to fight the module stack.
            raise RuntimeError(
                "The `datasets` package is unavailable (%s), so the gated ImageNet download "
                "cannot run.\n"
                "Three ways forward:\n"
                "  1. Alliance clusters: `datasets` needs pyarrow from the arrow MODULE. In a\n"
                "     clean shell run, IN THIS ORDER:\n"
                "         module --force purge && module load StdEnv/2023 python/3.11 gcc arrow\n"
                "         source <your-venv>/bin/activate\n"
                "         pip install --no-index datasets\n"
                "     The arrow module must be loaded BEFORE the venv is activated.\n"
                "  2. Build the cache on any machine that has `datasets` (Colab, a laptop) and\n"
                "     copy the single file it produces:\n"
                "         %s\n"
                "     It is a few hundred KB, self-contained and architecture-independent.\n"
                "  3. Skip the gated dataset entirely with data.source: local_folder and a\n"
                "     directory of <classid>_<name>.png images (see the README)."
                % (exc, cache)) from exc
        wanted = IMAGENET_EXAMPLES[:n]
        target = {c for c, _ in wanted}
        print("Streaming ImageNet validation images for %d classes ..." % len(target))
        ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=token)
        collected: Dict[int, Any] = {}
        for sample in ds:
            cls = sample["label"]
            if cls in target and cls not in collected:
                collected[cls] = sample["image"].convert("RGB")
            if len(collected) == len(target):
                break
        missing = target - set(collected)
        if missing:
            raise RuntimeError("No validation image found for classes %s" % sorted(missing))
        images = np.stack([to_float(np.array(center_crop(collected[c], self.image_size)))
                           for c, _ in wanted], axis=0).astype(np.float32)
        labels = np.array([c for c, _ in wanted], np.int32)
        names = [nm for _, nm in wanted]
        np.savez_compressed(cache, images=images, labels=labels,
                            names=np.array(names, dtype=object))
        return SourceExamples(images, labels, names, "hf_imagenet_val", {"cache": str(cache)})

    def _load_local(self, n: int, folder: str) -> SourceExamples:
        root = Path(folder)
        if not root.is_dir():
            raise RuntimeError("data.local_folder does not exist: %s" % folder)
        files = sorted(p for p in root.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))[:n]
        if not files:
            raise RuntimeError("No images found in %s" % folder)
        label_map: Dict[str, int] = {}
        lj = root / "labels.json"
        if lj.exists():
            label_map = {str(k): int(v) for k, v in json.loads(lj.read_text()).items()}
        images, labels, names = [], [], []
        for p in files:
            images.append(to_float(np.array(center_crop(Image.open(p).convert("RGB"),
                                                        self.image_size))))
            if p.name in label_map:
                labels.append(label_map[p.name])
            elif p.stem.split("_")[0].isdigit():
                labels.append(int(p.stem.split("_")[0]))
            else:
                raise RuntimeError("No label for %s. Class conditioning needs a true ImageNet "
                                   "label: provide labels.json {filename: class_id} in %s, or "
                                   "name files '<classid>_<n>.png'." % (p.name, root))
            names.append(p.stem)
        return SourceExamples(np.stack(images).astype(np.float32), np.array(labels, np.int32),
                              names, "local_folder", {"folder": str(root)})
