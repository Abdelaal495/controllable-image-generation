#!/usr/bin/env python3

"""
Extract the already-frozen ImageNet-100 benchmark from the original
ILSVRC ImageNet validation tarball.

STANDARD LIBRARY ONLY.

No datasets.
No pyarrow.
No numpy.
No torch.
No jax.
No repo imports.
"""

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--selection-plan", required=True)
    p.add_argument("--class-index", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    archive = Path(args.archive)
    plan_path = Path(args.selection_plan)
    class_index_path = Path(args.class_index)
    out = Path(args.output)

    if not archive.is_file():
        raise RuntimeError(f"Archive not found: {archive}")

    if not plan_path.is_file():
        raise RuntimeError(f"Selection plan not found: {plan_path}")

    out.mkdir(parents=True, exist_ok=True)

    # Refuse to mix with an existing benchmark.
    existing_images = [
        p for p in out.iterdir()
        if p.suffix.lower() in (".jpeg", ".jpg", ".png", ".webp", ".bmp")
    ]
    if existing_images:
        raise RuntimeError(
            f"{out} already contains {len(existing_images)} images. "
            "Refusing to overwrite/mix benchmark data."
        )

    # ------------------------------------------------------------
    # Load the exact selection made by the previous seed-42/43 run.
    # ------------------------------------------------------------

    plan = json.loads(plan_path.read_text())

    selections = plan["selection"]

    if len(selections) != 100:
        raise RuntimeError(
            f"Expected 100 entries in selection plan, found {len(selections)}."
        )

    class_ids = [int(x["class_id"]) for x in selections]

    if len(set(class_ids)) != 100:
        raise RuntimeError("Selection plan does not contain 100 unique classes.")

    # ------------------------------------------------------------
    # Canonical ImageNet class ID -> [synset, human-readable name]
    # ------------------------------------------------------------

    class_index = json.loads(class_index_path.read_text())

    wanted = {}

    for item in selections:
        class_id = int(item["class_id"])
        desired_rank = int(item["within_class_validation_rank"])

        synset, class_name = class_index[str(class_id)]

        wanted[synset] = {
            "class_id": class_id,
            "class_name": class_name,
            "desired_rank": desired_rank,
        }

    if len(wanted) != 100:
        raise RuntimeError("Class-index mapping produced duplicate synsets.")

    print("Frozen benchmark:")
    print(f"  classes    : {len(wanted)}")
    print(f"  class seed : {plan.get('class_seed')}")
    print(f"  image seed : {plan.get('image_seed')}")
    print()
    print("Streaming validation archive...")

    # Count validation examples encountered for each chosen synset.
    seen = {synset: 0 for synset in wanted}
    extracted = {}

    # r|gz = sequential streaming access.
    # We never unpack the other ~49,900 images.
    with tarfile.open(archive, mode="r|gz") as tf:

        for member in tf:

            if not member.isfile():
                continue

            basename = Path(member.name).name

            if not basename.lower().endswith(".jpeg"):
                continue

            stem = Path(basename).stem

            # Original HF archive convention:
            #
            #   ..._<SYNSET>.JPEG
            #
            # e.g. final token may be n01440764.
            try:
                synset = stem.rsplit("_", 1)[1]
            except IndexError:
                continue

            if synset not in wanted:
                continue

            rank = seen[synset]
            target = wanted[synset]

            if rank == target["desired_rank"]:

                src = tf.extractfile(member)

                if src is None:
                    raise RuntimeError(
                        f"Could not read archive member {member.name}"
                    )

                class_id = target["class_id"]

                # Prefix with the numeric ImageNet class so that the repo's
                # local_folder loader has a deterministic ascending order.
                filename = f"{class_id:03d}_{basename}"
                dst = out / filename

                with open(dst, "wb") as f:
                    shutil.copyfileobj(src, f)

                extracted[synset] = {
                    "filename": filename,
                    "class_id": class_id,
                    "synset": synset,
                    "class_name": target["class_name"],
                    "within_class_validation_rank": rank,
                    "original_archive_member": member.name,
                }

                print(
                    f"[{len(extracted):3d}/100] "
                    f"class={class_id:03d} "
                    f"synset={synset} "
                    f"rank={rank:02d} "
                    f"{target['class_name']}"
                )

                if len(extracted) == 100:
                    break

            seen[synset] += 1

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    if len(extracted) != 100:
        missing = sorted(
            (wanted[s]["class_id"], s, wanted[s]["desired_rank"])
            for s in wanted
            if s not in extracted
        )
        raise RuntimeError(
            f"Only extracted {len(extracted)}/100 images.\n"
            f"Missing: {missing}"
        )

    # Write everything in numeric class-ID order.
    records = sorted(
        extracted.values(),
        key=lambda x: x["class_id"]
    )

    labels = {
        r["filename"]: r["class_id"]
        for r in records
    }

    (out / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n"
    )

    for r in records:
        r["sha256"] = sha256(out / r["filename"])

    manifest = {
        "source": "ILSVRC/imagenet-1k",
        "source_split": "validation",
        "source_archive": "val_images.tar.gz",
        "selection_plan": plan_path.name,
        "class_seed": plan.get("class_seed"),
        "image_seed": plan.get("image_seed"),
        "num_images": 100,
        "images": records,
    }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    with open(out / "manifest.csv", "w", newline="") as f:
        fields = [
            "filename",
            "class_id",
            "synset",
            "class_name",
            "within_class_validation_rank",
            "original_archive_member",
            "sha256",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    with open(out / "SHA256SUMS.txt", "w") as f:
        for r in records:
            f.write(f"{r['sha256']}  {r['filename']}\n")

    image_files = [
        p for p in out.iterdir()
        if p.suffix.lower() in (".jpeg", ".jpg")
    ]

    assert len(image_files) == 100
    assert len(labels) == 100
    assert len(set(labels.values())) == 100

    print()
    print("=" * 72)
    print("IMAGENET-100 BENCHMARK READY")
    print("=" * 72)
    print(f"Output         : {out}")
    print(f"Images         : {len(image_files)}")
    print(f"Unique classes : {len(set(labels.values()))}")
    print(f"Class seed     : {plan.get('class_seed')}")
    print(f"Image seed     : {plan.get('image_seed')}")
    print(f"Manifest       : {out / 'manifest.json'}")
    print(f"Checksums      : {out / 'SHA256SUMS.txt'}")


if __name__ == "__main__":
    main()
