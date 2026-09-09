"""Build the `data.source: local_folder` pool that runs larger than 32 images need.

`src.data.IMAGENET_EXAMPLES` holds 32 curated classes and `data.source: hf_imagenet_val`
refuses anything beyond it, so any run with `num_images > 32` has to come from a local
folder.  This writes one folder that `src.data.DataManager._load_local` can read directly:
one validation image per class, named so that sorted order is the pool order, plus the
`labels.json` that carries each file's true ImageNet class id.

Two sources, in order of preference:
  * ILSVRC/imagenet-1k, if HF_TOKEN is set and the licence has been accepted;
  * evanarlian/imagenet_1k_resized_256 otherwise -- an ungated mirror of the same
    validation split whose `label` feature uses the identical 1000-class ordering
    (verified against the curated class names in src/data.py).

    python build_local_imagenet_pool.py                        # the 32 curated classes
    python build_local_imagenet_pool.py --num-classes 100 --seed 0
    python build_local_imagenet_pool.py --frozen-manifest benchmarks/imagenet100_c42_i43/manifest.csv

The third form rebuilds the FROZEN paper benchmark (upstream's 100 images, class seed 42 /
image seed 43) from the ungated mirror without the gated originals.  The mirror is sorted by
label with exactly 50 validation images per class, and within each class its rows run in
REVERSED validation-filename order, so upstream's `within_class_validation_rank` r maps to
mirror row  class_id * 50 + (49 - r).  Verified by content fingerprint on 2026-09-06: with the
paper's operators the degraded deblur / 2x-SR PSNR come out 25.91 / 22.81 against the paper's
25.97 / 22.80 (the forward mapping gives 26.39 / 23.29, a random 100-class pool 25.52 / 22.40).
The residual ~0.05 dB is the mirror's JPEG re-encoding at short side 256 before our crop.

WHICH classes are chosen is a property of the pool, not of the experiment: `runtime.seed`
in the configuration files controls generative noise, measurement noise, masks and stroke
geometry, and never the choice of source images.  `--seed` here is therefore a separate
knob, and the manifest written alongside the images records it so a pool can be rebuilt
exactly.
"""
import argparse, io, json, random, sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import IMAGENET_EXAMPLES, center_crop

MIRROR = "evanarlian/imagenet_1k_resized_256"
VAL_FILES = ("data/val-00000-of-00002-b5248be478d25e41.parquet",
             "data/val-00001-of-00002-85f3d9c8fa1edb63.parquet")


def choose(num_classes, seed):
    """The class ids to collect, and a readable name for each.

    The curated 32 always come first and in their original order, so a larger pool is a
    superset of the smaller one and the two share their first 32 image ids.  Anything
    beyond that is drawn from the remaining 968 classes with `seed`.
    """
    curated = [(c, n) for c, n in IMAGENET_EXAMPLES]
    if num_classes <= len(curated):
        return curated[:num_classes]
    rest = sorted(set(range(1000)) - {c for c, _ in curated})
    random.Random(seed).shuffle(rest)
    extra = [(c, "class%03d" % c) for c in rest[:num_classes - len(curated)]]
    return curated + extra


def build_frozen(manifest: Path, out: Path) -> None:
    """Upstream's frozen 100 from the mirror: mirror_row = class_id*50 + (49 - rank)."""
    import csv
    rows = list(csv.DictReader(open(manifest)))
    want = {}                                   # mirror row -> (filename stem, class id, record)
    for r in rows:
        cls, rank = int(r["class_id"]), int(r["within_class_validation_rank"])
        want[cls * 50 + (49 - rank)] = (Path(r["filename"]).stem, cls, r)
    out.mkdir(parents=True, exist_ok=True)
    labels, records, offset = {}, [], 0
    for f in VAL_FILES:
        pf = pq.ParquetFile(hf_hub_download(MIRROR, f, repo_type="dataset"))
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            need = [i for i in want if offset <= i < offset + n]
            if need:
                tbl = pf.read_row_group(rg, columns=["image", "label"])
                imgs, labs = tbl.column("image"), tbl.column("label").to_pylist()
                for i in need:
                    stem, cls, r = want[i]
                    assert labs[i - offset] == cls, (i, labs[i - offset], cls)
                    # stored as the mirror stores it (short side 256); the loader crops
                    Image.open(io.BytesIO(imgs[i - offset]["bytes"].as_py())).convert("RGB").save(out / (stem + ".png"))
                    labels[stem + ".png"] = cls
                    records.append({"filename": stem + ".png", "class_id": cls, "class_name": r["class_name"],
                                    "within_class_validation_rank": int(r["within_class_validation_rank"]),
                                    "mirror_row": i, "upstream_original": r["original_archive_member"],
                                    "upstream_sha256_original": r["sha256"]})
            offset += n
    missing = sorted(set(want) - {rec["mirror_row"] for rec in records})
    if missing:
        raise SystemExit("mirror rows not found: %s" % missing[:5])
    records.sort(key=lambda rec: rec["filename"])
    (out / "labels.json").write_text(json.dumps(labels, indent=1, sort_keys=True))
    (out / "pool_manifest.json").write_text(json.dumps({
        "reproduces": "%s (upstream: ILSVRC/imagenet-1k val, class_seed 42, image_seed 43)" % manifest,
        "source": "%s val parquet, sorted by label with 50 per class" % MIRROR,
        "mapping": "mirror_row = class_id*50 + (49 - within_class_validation_rank)",
        "evidence": ("the mirror's per-class blocks run in REVERSED validation-filename order; content-"
                     "fingerprint check with the paper's operators: degraded deblur / SR PSNR 25.91 / 22.81 "
                     "vs the paper's 25.97 / 22.80 (forward mapping 26.39 / 23.29)."),
        "images": records}, indent=1))
    print("wrote %d frozen-benchmark image(s) + labels.json + pool_manifest.json to %s" % (len(labels), out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-manifest", default=None,
                    help="rebuild upstream's frozen benchmark from this manifest.csv instead of "
                         "drawing classes (default out: cache/data/imagenet100_c42_i43_mirror)")
    ap.add_argument("--num-classes", type=int, default=len(IMAGENET_EXAMPLES))
    ap.add_argument("--seed", type=int, default=0,
                    help="only affects classes beyond the curated 32")
    ap.add_argument("--out", default=None, help="default: cache/data/imagenet_val_<N>")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    if args.frozen_manifest:
        build_frozen(Path(args.frozen_manifest),
                     Path(args.out) if args.out else root / "cache" / "data" / "imagenet100_c42_i43_mirror")
        return
    out = Path(args.out) if args.out else root / "cache" / "data" / (
        "imagenet_val_local" if args.num_classes <= len(IMAGENET_EXAMPLES)
        else "imagenet_val_%d" % args.num_classes)
    out.mkdir(parents=True, exist_ok=True)

    wanted = {c: (i, name) for i, (c, name) in enumerate(choose(args.num_classes, args.seed))}
    print("collecting %d class(es) into %s" % (len(wanted), out))

    found = {}
    for f in VAL_FILES:
        if len(found) == len(wanted):
            break
        path = hf_hub_download(MIRROR, f, repo_type="dataset")
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            if len(found) == len(wanted):
                break
            tbl = pf.read_row_group(rg, columns=["image", "label"])
            for row, cls in enumerate(tbl.column("label").to_pylist()):
                if cls in wanted and cls not in found:
                    found[cls] = tbl.column("image")[row]["bytes"].as_py()
        print("  %s -> %d/%d" % (Path(f).name, len(found), len(wanted)), flush=True)

    missing = sorted(set(wanted) - set(found))
    if missing:
        raise SystemExit("no validation image for classes %s" % missing)

    labels = {}
    width = max(2, len(str(len(wanted) - 1)))
    for cls, (ordinal, name) in sorted(wanted.items(), key=lambda kv: kv[1][0]):
        fname = "%0*d_%s.png" % (width, ordinal, name)
        center_crop(Image.open(io.BytesIO(found[cls])).convert("RGB"), 256).save(out / fname)
        labels[fname] = int(cls)
    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    (out / "pool_manifest.json").write_text(json.dumps(
        {"num_classes": args.num_classes, "seed": args.seed, "mirror": MIRROR,
         "curated_prefix": len(IMAGENET_EXAMPLES),
         "classes": [int(c) for c, _ in choose(args.num_classes, args.seed)]}, indent=2))
    print("wrote %d image(s) + labels.json + pool_manifest.json" % len(labels))


if __name__ == "__main__":
    main()
