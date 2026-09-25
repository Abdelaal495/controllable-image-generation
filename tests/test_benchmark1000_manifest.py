"""The 1000-image frozen benchmark: one validation image per ImageNet-1k class.

Checks benchmarks/imagenet1000_c42_i43 without a dataset, a token, a GPU or the network:

  1. 1000 manifest rows and 1000 plan entries, agreeing row for row, seeds 42 / 43;
  2. every class 0..999 exactly once, every rank in 0..49, every mirror row distinct and in
     the block of its own class;
  3. no selected row is in the TUNING pool (the seed-0 cache/data/imagenet_val_100 the
     hyperparameter search ran on), and the plan records exactly those excluded rows and
     every re-draw they forced;
  4. filenames are unique and encode the class, synset and rank they stand for; synset and
     class name agree with the class index the 100-image benchmark was built with;
  5. the columns are the 100-image manifest's columns, and no filename -- hence no image
     id -- is shared with the 100-image benchmark;
  6. the draw is reproducible: scripts/make_frozen_selection.draw() regenerates the plan,
     and with 100 classes and no exclusion it regenerates benchmarks/imagenet100_c42_i43,
     so the two benchmarks share one selection procedure; the script's view of the tuning
     pool agrees with an independent transcription;
  7. nothing here reached for the network: no download-capable module was imported.

    python tests/test_benchmark1000_manifest.py
"""
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Nothing below should even try to download; a regression must fail rather than fetch.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

BENCH = REPO_ROOT / "benchmarks" / "imagenet1000_c42_i43"
BENCH100 = REPO_ROOT / "benchmarks" / "imagenet100_c42_i43"
CLASS_INDEX = REPO_ROOT / "benchmarks" / "imagenet_class_index.json"
TUNING_POOL = REPO_ROOT / "cache" / "data" / "imagenet_val_100"
SELECTION_SCRIPT = REPO_ROOT / "scripts" / "make_frozen_selection.py"
IMAGES_PER_CLASS = 50
COLUMNS = ["filename", "class_id", "synset", "class_name", "within_class_validation_rank",
           "original_archive_member", "sha256"]

PASSED = [0]
FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print("  [%s] %-54s %s" % ("PASS" if ok else "FAIL", name, detail))
    if ok:
        PASSED[0] += 1
    else:
        FAILED.append(name)


def mirror_row(class_id: int, rank: int) -> int:
    return class_id * IMAGES_PER_CLASS + (IMAGES_PER_CLASS - 1 - rank)


def load():
    with open(BENCH / "manifest.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames)
    plan = json.loads((BENCH / "selection_plan.json").read_text())
    with open(BENCH100 / "manifest.csv", newline="") as f:
        rows100 = list(csv.DictReader(f))
    plan100 = json.loads((BENCH100 / "selection_plan.json").read_text())
    index = {int(k): tuple(v) for k, v in json.loads(CLASS_INDEX.read_text()).items()}
    return rows, columns, plan, rows100, plan100, index


def tuning_pool_rows():
    """(class_id, mirror_row) of the tuning pool: from its manifest when a copy exists, else
    by an INDEPENDENT transcription of build_local_imagenet_pool.choose(100, seed=0)."""
    manifest = TUNING_POOL / "pool_manifest.json"
    if manifest.is_file():
        classes = [int(c) for c in json.loads(manifest.read_text())["classes"]]
        how = "from %s" % manifest.relative_to(REPO_ROOT).as_posix()
    else:
        from src.data import IMAGENET_EXAMPLES
        curated = [c for c, _ in IMAGENET_EXAMPLES]
        rest = sorted(set(range(1000)) - set(curated))
        random.Random(0).shuffle(rest)
        classes = curated + rest[:100 - len(curated)]
        how = "reconstructed: the curated 32, then 68 of the rest shuffled by random.Random(0)"
    # the seeded builder keeps the FIRST row the mirror yields per class: class_id*50, rank 49
    return {(c, c * IMAGES_PER_CLASS) for c in classes}, how


def load_selection_script():
    spec = importlib.util.spec_from_file_location("make_frozen_selection", SELECTION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =====================================================================================
# 1. sizes and agreement between manifest and plan
# =====================================================================================
def test_sizes(rows, plan):
    print("\n1. 1000 rows, 1000 plan entries, seeds 42 / 43, row-for-row agreement")
    check("manifest has 1000 rows", len(rows) == 1000, "%d rows" % len(rows))
    check("plan has 1000 selection entries", len(plan["selection"]) == 1000)
    check("plan declares num_classes 1000", plan.get("num_classes") == 1000)
    check("class seed 42, image seed 43",
          (plan.get("class_seed"), plan.get("image_seed")) == (42, 43),
          "%r / %r" % (plan.get("class_seed"), plan.get("image_seed")))
    check("dataset / split / image_size as the 100-image plan",
          (plan.get("dataset"), plan.get("split"), plan.get("image_size"))
          == ("ILSVRC/imagenet-1k", "validation", 256))
    pairs_csv = [(int(r["class_id"]), int(r["within_class_validation_rank"])) for r in rows]
    pairs_plan = [(int(s["class_id"]), int(s["within_class_validation_rank"]))
                  for s in plan["selection"]]
    check("manifest and plan list the same (class, rank) in the same order",
          pairs_csv == pairs_plan)
    return pairs_csv


# =====================================================================================
# 2. coverage: every class once, ranks in range, mirror rows distinct
# =====================================================================================
def test_coverage(pairs):
    print("\n2. every class exactly once, ranks in 0..49, mirror rows distinct")
    classes = [c for c, _ in pairs]
    check("class ids are exactly 0..999, each once", sorted(classes) == list(range(1000)))
    check("rows are in ascending class order (sorted(files) is the pool order)",
          classes == sorted(classes))
    ranks = [r for _, r in pairs]
    check("every rank in 0..49", all(0 <= r < IMAGES_PER_CLASS for r in ranks),
          "min %d max %d" % (min(ranks), max(ranks)))
    check("ranks are not all one value (a real draw)", len(set(ranks)) > 1,
          "%d distinct ranks" % len(set(ranks)))
    rows_m = [mirror_row(c, r) for c, r in pairs]
    check("mirror rows distinct", len(set(rows_m)) == len(rows_m))
    check("every mirror row inside its class block of 50",
          all(m // IMAGES_PER_CLASS == c for m, (c, _) in zip(rows_m, pairs)))
    check("mirror rows inside the 50 000-row validation split",
          all(0 <= m < 1000 * IMAGES_PER_CLASS for m in rows_m))


# =====================================================================================
# 3. disjoint from the tuning pool, and the plan says so
# =====================================================================================
def test_tuning_pool(pairs, plan):
    print("\n3. disjoint from the tuning pool (seed-0 imagenet_val_100), recorded in the plan")
    tuning, how = tuning_pool_rows()
    print("     tuning pool rows %s" % how)
    check("tuning pool has 100 rows", len(tuning) == 100)
    selected = {(c, mirror_row(c, r)) for c, r in pairs}
    overlap = sorted(selected & tuning)
    check("no selected (class, mirror row) is in the tuning pool", not overlap,
          "overlap: %s" % overlap[:5] if overlap else "")
    tuning_classes = {c for c, _ in tuning}
    check("no tuning class is selected at rank 49 (its tuning row)",
          all(r != 49 for c, r in pairs if c in tuning_classes))
    ex = plan.get("tuning_pool_exclusion", {})
    recorded = {(int(e["class_id"]), int(e["mirror_row"])) for e in ex.get("excluded_rows", [])}
    check("plan records exactly the excluded tuning rows", recorded == tuning,
          "%d recorded" % len(recorded))
    check("every recorded excluded row carries rank 49",
          all(int(e["within_class_validation_rank"]) == 49 for e in ex.get("excluded_rows", [])))
    check("plan names the exclusion source", bool(ex.get("source")), str(ex.get("source"))[:80])
    redraws = ex.get("redraws", [])
    final = dict(pairs)
    check("every re-draw is a tuning class that first hit its tuning row",
          all(int(d["class_id"]) in tuning_classes and d["rejected_ranks"]
              and all(int(x) == 49 for x in d["rejected_ranks"]) for d in redraws),
          "%d re-draw(s): %s" % (len(redraws), [d["class_id"] for d in redraws]))
    check("every re-draw's final rank is the manifest's rank",
          all(final[int(d["class_id"])] == int(d["within_class_validation_rank"])
              for d in redraws))


# =====================================================================================
# 4. filenames and the class index
# =====================================================================================
def test_filenames(rows, index):
    print("\n4. filenames unique and self-describing; synset / class name from the class index")
    names = [r["filename"] for r in rows]
    check("filenames unique", len(set(names)) == len(names))
    check("class index has 1000 entries with unique synsets",
          len(index) == 1000 and len({s for s, _ in index.values()}) == 1000)
    check("filename == <class:03d>_<synset>_r<rank:02d>.JPEG for every row",
          all(r["filename"] == "%03d_%s_r%02d.JPEG" % (int(r["class_id"]), r["synset"],
                                                      int(r["within_class_validation_rank"]))
              for r in rows))
    check("synset and class_name match the class index for every row",
          all((r["synset"], r["class_name"]) == index[int(r["class_id"])] for r in rows))
    check("filename stems sort in class order (the loader's sorted(files)[:n])",
          [Path(n).stem for n in names] == sorted(Path(n).stem for n in names))


# =====================================================================================
# 5. schema shared with the 100-image benchmark, ids disjoint from it
# =====================================================================================
def test_schema(columns, rows, rows100, plan100):
    print("\n5. the 100-image manifest's columns; no filename shared with the 100-image benchmark")
    check("columns are exactly the 100-image manifest's", columns == COLUMNS, str(columns))
    with open(BENCH100 / "manifest.csv", newline="") as f:
        columns100 = list(csv.DictReader(f).fieldnames)
    check("100-image manifest still has those columns (unchanged)", columns100 == COLUMNS)
    check("100-image plan still has 100 entries (unchanged)", len(plan100["selection"]) == 100)
    shared = {r["filename"] for r in rows} & {r["filename"] for r in rows100}
    check("no filename shared with the 100-image benchmark", not shared, str(sorted(shared))[:80])
    check("original_archive_member / sha256 need the gated archive: blank, and the plan says why",
          all(r["original_archive_member"] == "" and r["sha256"] == "" for r in rows))


# =====================================================================================
# 6. the draw is reproducible and shared with the 100-image benchmark
# =====================================================================================
def test_reproducible(pairs, plan, plan100):
    print("\n6. scripts/make_frozen_selection.draw regenerates both benchmarks' plans")
    mfs = load_selection_script()
    tuning, _ = tuning_pool_rows()
    script_rows, source = mfs.tuning_pool_rows(TUNING_POOL)
    check("the script's tuning-pool rows agree with this test's transcription",
          set(script_rows) == tuning, source[:70])
    selection, redraws = mfs.draw(1000, 42, 43, tuning)
    check("draw(1000, 42, 43, tuning rows) == the committed 1000-image selection",
          selection == pairs)
    check("its re-draws == the plan's recorded re-draws",
          redraws == plan.get("tuning_pool_exclusion", {}).get("redraws"))
    selection100, redraws100 = mfs.draw(100, 42, 43, ())
    expected100 = [(int(s["class_id"]), int(s["within_class_validation_rank"]))
                   for s in plan100["selection"]]
    check("draw(100, 42, 43) == benchmarks/imagenet100_c42_i43 (one procedure, two sizes)",
          selection100 == expected100 and not redraws100)
    other, _ = mfs.draw(1000, 42, 44, tuning)
    check("a different image seed changes the ranks", other != selection)


# =====================================================================================
# 7. nothing touched the network
# =====================================================================================
def test_no_download():
    print("\n7. no download-capable module was imported")
    loaded = [m for m in ("huggingface_hub", "datasets", "pyarrow", "requests", "urllib3")
              if m in sys.modules]
    check("huggingface_hub / datasets / pyarrow / requests never imported", not loaded,
          str(loaded))
    check("HF offline switches were set for the whole run",
          os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("HF_DATASETS_OFFLINE") == "1")


def main() -> int:
    print("=" * 94)
    print("The 1000-image frozen benchmark: benchmarks/imagenet1000_c42_i43")
    print("=" * 94)
    rows, columns, plan, rows100, plan100, index = load()
    pairs = test_sizes(rows, plan)
    test_coverage(pairs)
    test_tuning_pool(pairs, plan)
    test_filenames(rows, index)
    test_schema(columns, rows, rows100, plan100)
    test_reproducible(pairs, plan, plan100)
    test_no_download()

    total = PASSED[0] + len(FAILED)
    print("\n%d/%d checks passed" % (PASSED[0], total))
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
