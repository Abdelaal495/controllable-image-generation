#!/usr/bin/env python3

import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image


def save_uint8(arr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(arr)

    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)

    Image.fromarray(x).save(path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    benchmark_dir = Path(args.benchmark_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Read aggregate results so we can recover naming / steps
    # -----------------------------
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise RuntimeError(f"Missing {results_csv}")

    rows = []
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    rows_by_job = {r["job_id"]: r for r in rows}

    # -----------------------------
    # Export originals once
    # -----------------------------
    benchmark_images = sorted(
        [p for p in benchmark_dir.iterdir()
         if p.suffix.lower() in [".jpeg", ".jpg", ".png", ".webp", ".bmp"]]
    )

    print(f"Found {len(benchmark_images)} benchmark images")

    for i, img_path in enumerate(benchmark_images):
        out_path = output_dir / "by_image" / f"image_{i:03d}" / "original.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # keep it simple: load and re-save
        Image.open(img_path).convert("RGB").save(out_path)

    # -----------------------------
    # Export reconstructions
    # -----------------------------
    meta_paths = sorted(run_dir.glob("**/metadata.json"))
    print(f"Found {len(meta_paths)} metadata files")

    exported = 0

    for meta_path in meta_paths:
        job_dir = meta_path.parent
        npz_path = job_dir / "results.npz"

        if not npz_path.exists():
            print(f"Skipping {job_dir} (missing results.npz)")
            continue

        meta = json.loads(meta_path.read_text())
        job_id = meta.get("job_id")

        row = rows_by_job.get(job_id, {})

        experiment = row.get("experiment", meta.get("experiment", "unknown_experiment"))
        model = row.get("model", meta.get("model", "unknown_model"))
        method = row.get("method", meta.get("method", "unknown_method"))

        # Distinguish SDEdit step settings
        filename = method
        if method == "sdedit":
            steps = row.get("steps")
            if steps:
                filename += f"_steps{steps}"

        filename += ".png"

        z = np.load(npz_path, allow_pickle=True)
        if "reconstruction" not in z.files:
            print(f"Skipping {npz_path} (no 'reconstruction' key)")
            continue

        recon = z["reconstruction"]   # (100,256,256,3) uint8
        if recon.shape[0] != 100:
            print(f"Warning: expected 100 images, found {recon.shape[0]} in {npz_path}")

        for i in range(recon.shape[0]):
            out_path = (
                output_dir / "by_image" / f"image_{i:03d}" / experiment / model / filename
            )
            save_uint8(recon[i], out_path)

        exported += 1
        print(f"[{exported:3d}] {experiment} | {model} | {filename}")

    print()
    print("Done.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
