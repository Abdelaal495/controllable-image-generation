#!/usr/bin/env python
"""Draw a frozen ImageNet-1k validation benchmark: one image per class, as (class, rank) pairs.

The selection step only.  It needs no dataset, no token and no network: it writes the PLAN
(which validation image of which class) and the manifest that
`scripts/build_local_imagenet_pool.py --frozen-manifest` turns into an image folder from the
ungated mirror, one row at a time, through mirror_row = class_id*50 + (49 - rank).

    python scripts/make_frozen_selection.py                     # 1000 classes, seeds 42 / 43
        -> benchmarks/imagenet1000_c42_i43/selection_plan.json + manifest.csv
    python scripts/build_local_imagenet_pool.py \\
        --frozen-manifest benchmarks/imagenet1000_c42_i43/manifest.csv --out cache/data/imagenet1000_c42_i43_mirror

The draw is the one that produced benchmarks/imagenet100_c42_i43 -- with --num-classes 100
and --no-exclude it reproduces that plan exactly, which tests/test_benchmark1000_manifest.py
checks:

    classes = sorted(numpy.random.default_rng(class_seed).choice(1000, num_classes, replace=False))
    ranks   = numpy.random.default_rng(image_seed).integers(0, 50, size=num_classes)   # in class order

plus one rule: a (class, mirror row) that the TUNING pool holds -- the seed-0
cache/data/imagenet_val_100 that scripts/hpo.py searched on -- is never selected; that
class's rank is drawn again from the same generator until it lands elsewhere.  The tuning
pool keeps the FIRST mirror row of each of its classes, i.e. rank 49, so only classes drawn
at 49 are touched.  Every excluded row and every re-draw is written into the plan.

The tuning pool's pool_manifest.json is read when it exists; otherwise its classes are
reconstructed by the rule build_local_imagenet_pool.choose(100, seed=0) uses (the curated
32, then 68 of the rest shuffled by random.Random(0)).  The pool is gitignored, so the
exclusion must not depend on a copy being around; when a copy IS around the two are
cross-checked.

Schema: the columns of benchmarks/imagenet100_c42_i43/manifest.csv.  `original_archive_member`
and `sha256` identify the file inside the gated ILSVRC archive; neither can be known without
that archive (its member order is not the validation-filename order), so both are blank and
`filename` carries the identity instead: <class:03d>_<synset>_r<rank:02d>.JPEG.  The mirror
build records the mirror row and the class name per image, exactly as it does for the 100.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data import IMAGENET_EXAMPLES  # noqa: E402  (numpy + Pillow only; nothing is fetched)

IMAGES_PER_CLASS = 50          # the ImageNet-1k validation split: 50 per class, 50 000 in all
NUM_CLASSES = 1000
DEFAULT_CLASS_INDEX = REPO / "benchmarks" / "imagenet_class_index.json"
DEFAULT_TUNING_POOL = REPO / "cache" / "data" / "imagenet_val_100"
TUNING_POOL_CLASSES, TUNING_POOL_SEED = 100, 0
MANIFEST_FIELDS = ["filename", "class_id", "synset", "class_name", "within_class_validation_rank",
                   "original_archive_member", "sha256"]


def mirror_row(class_id: int, rank: int) -> int:
    """The row of the ungated mirror that holds validation rank `rank` of `class_id`."""
    return class_id * IMAGES_PER_CLASS + (IMAGES_PER_CLASS - 1 - rank)


def rank_of(class_id: int, row: int) -> int:
    """Inverse of mirror_row."""
    return IMAGES_PER_CLASS - 1 - (row - class_id * IMAGES_PER_CLASS)


def shown(path: Path) -> str:
    """A path as the plan records it: relative to the repository when it lies inside it."""
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def tuning_pool_classes_by_rule(num_classes: int = TUNING_POOL_CLASSES,
                                seed: int = TUNING_POOL_SEED):
    """build_local_imagenet_pool.choose(num_classes, seed), transcribed.

    Not imported: that module needs pyarrow and huggingface_hub at import time and this
    script must run without either.  The curated 32 come first, then the rest shuffled.
    """
    curated = [c for c, _ in IMAGENET_EXAMPLES]
    rest = sorted(set(range(NUM_CLASSES)) - set(curated))
    random.Random(seed).shuffle(rest)
    return curated + rest[:num_classes - len(curated)]


def tuning_pool_rows(pool: Path):
    """The (class_id, mirror_row) pairs the tuning pool holds, and how that was established.

    A seeded pool keeps the FIRST row the mirror yields per class, class_id*50 + 0 -- rank
    49.  A frozen-style pool (built with --frozen-manifest) lists its mirror rows outright.
    """
    pool = Path(pool)
    manifest = pool / "pool_manifest.json" if pool.is_dir() else pool
    by_rule = [(c, c * IMAGES_PER_CLASS) for c in tuning_pool_classes_by_rule()]
    if not manifest.is_file():
        return by_rule, ("%s not found: classes reconstructed by the rule of "
                         "build_local_imagenet_pool.choose(%d, seed=%d), first mirror row per class"
                         % (shown(manifest), TUNING_POOL_CLASSES, TUNING_POOL_SEED))
    m = json.loads(manifest.read_text())
    if isinstance(m.get("images"), list):
        rows = [(int(r["class_id"]), int(r["mirror_row"])) for r in m["images"]]
        return rows, "read from %s (explicit mirror rows)" % shown(manifest)
    rows = [(int(c), int(c) * IMAGES_PER_CLASS) for c in m["classes"]]
    seeded = (int(m.get("num_classes", -1)), int(m.get("seed", -1)))
    if seeded == (TUNING_POOL_CLASSES, TUNING_POOL_SEED) and rows != by_rule:
        raise SystemExit("%s lists different classes from build_local_imagenet_pool.choose(%d, "
                         "seed=%d): the rule transcribed in this script is stale"
                         % (shown(manifest), TUNING_POOL_CLASSES, TUNING_POOL_SEED))
    return rows, ("read from %s (num_classes %s, seed %s), first mirror row per class"
                  % (shown(manifest), m.get("num_classes"), m.get("seed")))


def draw(num_classes: int, class_seed: int, image_seed: int, excluded=()):
    """The (class_id, rank) selection in class order, and the re-draws the exclusions forced."""
    excluded = set(excluded)
    classes = sorted(int(c) for c in np.random.default_rng(class_seed).choice(
        NUM_CLASSES, num_classes, replace=False))
    rng = np.random.default_rng(image_seed)
    ranks = [int(r) for r in rng.integers(0, IMAGES_PER_CLASS, size=num_classes)]
    redraws = []
    for i, c in enumerate(classes):
        if all((c, mirror_row(c, r)) in excluded for r in range(IMAGES_PER_CLASS)):
            raise SystemExit("every validation image of class %d is excluded" % c)
        rejected = []
        while (c, mirror_row(c, ranks[i])) in excluded:
            rejected.append(ranks[i])
            ranks[i] = int(rng.integers(0, IMAGES_PER_CLASS))
        if rejected:
            redraws.append({"class_id": c, "rejected_ranks": rejected,
                            "within_class_validation_rank": ranks[i]})
    return list(zip(classes, ranks)), redraws


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--class-seed", type=int, default=42)
    ap.add_argument("--image-seed", type=int, default=43)
    ap.add_argument("--tuning-pool", default=str(DEFAULT_TUNING_POOL),
                    help="pool folder (or its pool_manifest.json) whose rows are excluded; its "
                         "classes are reconstructed by rule when it is absent")
    ap.add_argument("--no-exclude", action="store_true",
                    help="draw without the tuning-pool exclusion (with --num-classes 100 this "
                         "reproduces benchmarks/imagenet100_c42_i43)")
    ap.add_argument("--class-index", default=str(DEFAULT_CLASS_INDEX),
                    help="class id -> [synset, name]; the file the 100-image benchmark used")
    ap.add_argument("--out", default=None,
                    help="default: benchmarks/imagenet<N>_c<class seed>_i<image seed>")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest (never one that has been run)")
    args = ap.parse_args()
    if not 1 <= args.num_classes <= NUM_CLASSES:
        raise SystemExit("--num-classes must be in 1..%d" % NUM_CLASSES)

    out = Path(args.out) if args.out else REPO / "benchmarks" / (
        "imagenet%d_c%d_i%d" % (args.num_classes, args.class_seed, args.image_seed))
    if (out / "manifest.csv").exists() and not args.force:
        raise SystemExit("%s already holds a manifest; a frozen benchmark is never rewritten "
                         "in place (pass --force only for one that has never been run)" % out)

    class_index = {int(k): v for k, v in json.loads(Path(args.class_index).read_text()).items()}
    if args.no_exclude:
        excluded, source = [], "none (--no-exclude)"
    else:
        excluded, source = tuning_pool_rows(Path(args.tuning_pool))
    selection, redraws = draw(args.num_classes, args.class_seed, args.image_seed, excluded)

    plan = {
        "dataset": "ILSVRC/imagenet-1k",
        "split": "validation",
        "num_classes": args.num_classes,
        "class_seed": args.class_seed,
        "image_seed": args.image_seed,
        "image_size": 256,
        "draw": {
            "classes": "sorted(numpy.random.default_rng(class_seed).choice(1000, num_classes, "
                       "replace=False))",
            "ranks": "numpy.random.default_rng(image_seed).integers(0, 50, size=num_classes), one "
                     "per class in class order; a rank whose (class, mirror_row) is excluded is "
                     "drawn again from the same generator",
            "mirror_row": "class_id*50 + (49 - within_class_validation_rank)",
            "same_as": "benchmarks/imagenet100_c42_i43 (reproduced exactly by --num-classes 100 "
                       "--no-exclude)",
        },
        "tuning_pool_exclusion": {
            "pool": None if args.no_exclude else shown(Path(args.tuning_pool)),
            "source": source,
            "excluded_rows": [{"class_id": c, "mirror_row": r,
                               "within_class_validation_rank": rank_of(c, r)}
                              for c, r in sorted(set(excluded))],
            "redraws": redraws,
        },
        "manifest_note": ("original_archive_member and sha256 are blank: both need the gated "
                          "ILSVRC archive, whose member order is not the validation-filename "
                          "order. filename encodes <class>_<synset>_r<rank>; "
                          "build_local_imagenet_pool.py records the mirror row per image."),
        "selection": [{"class_id": c, "within_class_validation_rank": r} for c, r in selection],
    }
    rows = []
    for c, r in selection:
        synset, name = class_index[c]
        rows.append({"filename": "%03d_%s_r%02d.JPEG" % (c, synset, r), "class_id": c,
                     "synset": synset, "class_name": name, "within_class_validation_rank": r,
                     "original_archive_member": "", "sha256": ""})

    out.mkdir(parents=True, exist_ok=True)
    (out / "selection_plan.json").write_text(json.dumps(plan, indent=2) + "\n", newline="\n")
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("wrote %s: %d classes, %d excluded tuning row(s), %d re-draw(s)%s"
          % (shown(out), len(rows), len(set(excluded)), len(redraws),
             (" for classes " + ", ".join(str(d["class_id"]) for d in redraws)) if redraws else ""))
    print("tuning pool: %s" % source)


if __name__ == "__main__":
    main()
