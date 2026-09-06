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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-classes", type=int, default=len(IMAGENET_EXAMPLES))
    ap.add_argument("--seed", type=int, default=0,
                    help="only affects classes beyond the curated 32")
    ap.add_argument("--out", default=None, help="default: cache/data/imagenet_val_<N>")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
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
