#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter


# ----------------------------
# configuration
# ----------------------------

IMAGE_SIZE = 256

TASKS = {
    "denoising_final100": {
        "type": "denoise",
        "sigma": 0.2,
        "seed": 1001,
    },
    "deblurring_final100": {
        "type": "deblur",
        "sigma": 0.05,
        "blur_sigma": 1.0,
        "seed": 1002,
    },
    "super_resolution_final100": {
        "type": "sr",
        "factor": 2,
        "sigma": 0.05,
        "seed": 1003,
    },
    "box_inpainting_final100": {
        "type": "box_inpaint",
        "box": 40,
        "sigma": 0.05,
        "seed": 1004,
    },
    "random_inpainting_final100": {
        "type": "random_inpaint",
        "missing_fraction": 0.7,
        "sigma": 0.01,
        "seed": 1005,
    },
}


# ----------------------------
# helpers
# ----------------------------

def get_bicubic():
    try:
        return Image.Resampling.BICUBIC
    except AttributeError:
        return Image.BICUBIC


BICUBIC = get_bicubic()


def center_crop_resize(img: Image.Image, size=256) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    img = img.crop((left, top, left + short, top + short))
    img = img.resize((size, size), BICUBIC)
    return img


def pil_to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32) / 255.0


def float_to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (255.0 * x).round().astype(np.uint8)


def save_float_image(x: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(float_to_uint8(x)).save(path)


def save_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img).save(path)


def add_gaussian_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32), 0.0, 1.0)


# ----------------------------
# corruption builders
# ----------------------------

def build_denoise(clean: np.ndarray, cfg, rng):
    corrupted = add_gaussian_noise(clean, cfg["sigma"], rng)
    extras = {}
    return corrupted, extras


def build_deblur(clean_img: Image.Image, cfg, rng):
    blurred = clean_img.filter(ImageFilter.GaussianBlur(radius=cfg["blur_sigma"]))
    blurred = pil_to_float(blurred)
    corrupted = add_gaussian_noise(blurred, cfg["sigma"], rng)
    extras = {}
    return corrupted, extras


def build_sr(clean_img: Image.Image, cfg, rng):
    factor = cfg["factor"]
    lr_size = IMAGE_SIZE // factor

    lr = clean_img.resize((lr_size, lr_size), BICUBIC)
    lr_float = pil_to_float(lr)
    lr_noisy = add_gaussian_noise(lr_float, cfg["sigma"], rng)

    # save a display-sized version for easy browsing
    lr_display = Image.fromarray(float_to_uint8(lr_noisy)).resize((IMAGE_SIZE, IMAGE_SIZE), BICUBIC)
    corrupted = pil_to_float(lr_display)

    extras = {
        "measurement_lr": lr_noisy,  # true low-res observed image
    }
    return corrupted, extras


def build_box_inpaint(clean: np.ndarray, cfg, rng):
    sigma = cfg["sigma"]
    box = cfg["box"]

    noisy = add_gaussian_noise(clean, sigma, rng)

    mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    start = (IMAGE_SIZE - box) // 2
    end = start + box
    mask[start:end, start:end] = 0

    corrupted = noisy.copy()
    corrupted[mask == 0] = 0.5  # gray missing area for display

    extras = {
        "mask": mask,
    }
    return corrupted, extras


def build_random_inpaint(clean: np.ndarray, cfg, rng):
    sigma = cfg["sigma"]
    missing_fraction = cfg["missing_fraction"]

    noisy = add_gaussian_noise(clean, sigma, rng)

    observed_mask = (rng.random((IMAGE_SIZE, IMAGE_SIZE)) > missing_fraction).astype(np.uint8)

    corrupted = noisy.copy()
    corrupted[observed_mask == 0] = 0.5  # gray missing area for display

    extras = {
        "mask": observed_mask,
    }
    return corrupted, extras


def build_corruption(clean_img: Image.Image, task_name: str, image_index: int):
    cfg = TASKS[task_name]
    rng = np.random.default_rng(cfg["seed"] + image_index)

    clean = pil_to_float(clean_img)

    if cfg["type"] == "denoise":
        return build_denoise(clean, cfg, rng)
    elif cfg["type"] == "deblur":
        return build_deblur(clean_img, cfg, rng)
    elif cfg["type"] == "sr":
        return build_sr(clean_img, cfg, rng)
    elif cfg["type"] == "box_inpaint":
        return build_box_inpaint(clean, cfg, rng)
    elif cfg["type"] == "random_inpaint":
        return build_random_inpaint(clean, cfg, rng)
    else:
        raise ValueError(f"Unknown task type: {cfg['type']}")


# ----------------------------
# main
# ----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--bundle-dir", required=True)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).resolve()
    bundle_dir = Path(args.bundle_dir).resolve()

    image_paths = sorted([
        p for p in benchmark_dir.iterdir()
        if p.suffix.lower() in [".jpeg", ".jpg", ".png", ".bmp", ".webp"]
    ])

    if len(image_paths) != 100:
        print(f"Warning: expected 100 benchmark images, found {len(image_paths)}")

    by_image_dir = bundle_dir / "by_image"
    if not by_image_dir.exists():
        raise RuntimeError(f"Expected existing bundle directory at {by_image_dir}")

    print(f"Found {len(image_paths)} benchmark images")
    print(f"Writing corruptions into: {bundle_dir}")
    print()

    for i, img_path in enumerate(image_paths):
        image_dir = by_image_dir / f"image_{i:03d}"
        image_dir.mkdir(parents=True, exist_ok=True)

        clean_img = center_crop_resize(Image.open(img_path), size=IMAGE_SIZE)

        # Save the actual 256x256 clean image used by the model
        clean_path = image_dir / "original_cropped.png"
        if not clean_path.exists():
            clean_img.save(clean_path)

        for task_name in TASKS:
            task_dir = image_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            corrupted, extras = build_corruption(clean_img, task_name, i)
            save_float_image(corrupted, task_dir / "corrupted.png")

            if "mask" in extras:
                save_mask(extras["mask"], task_dir / "mask.png")

            if "measurement_lr" in extras:
                save_float_image(extras["measurement_lr"], task_dir / "measurement_lr.png")

        print(f"[{i+1:3d}/100] image_{i:03d}")

    print()
    print("=" * 70)
    print("CORRUPTED-IMAGE EXPORT COMPLETE")
    print("=" * 70)
    print(f"Bundle directory: {bundle_dir}")


if __name__ == "__main__":
    main()
