from pathlib import Path
import csv
import random
import shutil
from collections import defaultdict, Counter
from tqdm import tqdm


DATASET_ROOT = Path(r"/root/workspace/Dataset/TSL")

ANNO_ROOT = DATASET_ROOT / "annotations" / "manual"
TOTAL_CORPUS = ANNO_ROOT / "total.corpus.csv"

FRAME_ROOT = DATASET_ROOT / "features" / "fullFrame-640x480px"
TOTAL_FRAME_ROOT = FRAME_ROOT / "total"

SPLIT_RATIO = {
    "train": 0.6,
    "dev": 0.2,
    "test": 0.2,
}

SEED = 0
LABEL_FIELD = "zh_gloss"
FRAME_EXT = ".png"


def ask_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()

        if answer == "y":
            return True

        if answer == "n":
            return False

        print("Invalid input. Please enter 'y' or 'n'.")


def read_total_corpus(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cannot find total corpus: {path}")

    rows = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        expected_header = "id|folder|signer|sentence|zh_sentence|zh_gloss"

        if len(header) != 1 or header[0] != expected_header:
            raise ValueError(
                f"Unexpected header: {header}\n"
                f"Expected single-column header: {expected_header}"
            )

        for line_idx, row in enumerate(reader, start=2):
            if not row:
                continue

            parts = row[0].split("|")

            if len(parts) != 6:
                raise ValueError(
                    f"Invalid row at line {line_idx}: {row[0]}\n"
                    f"Expected 6 fields, got {len(parts)}"
                )

            sample_id, folder, signer, sentence, zh_sentence, zh_gloss = parts

            rows.append({
                "id": sample_id,
                "folder": folder,
                "signer": signer,
                "sentence": sentence,
                "zh_sentence": zh_sentence,
                "zh_gloss": zh_gloss,
            })

    return rows


def stratified_split_by_sentence(rows, ratio, seed=42):
    rng = random.Random(seed)

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["sentence"]].append(row)

    split_rows = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for sentence, items in tqdm(
        sorted(grouped.items()),
        desc="Splitting by sentence"
    ):
        items = items[:]
        rng.shuffle(items)

        n = len(items)

        n_train = round(n * ratio["train"])
        n_dev = round(n * ratio["dev"])

        if n_train + n_dev > n:
            n_dev = n - n_train

        n_test = n - n_train - n_dev

        train_items = items[:n_train]
        dev_items = items[n_train:n_train + n_dev]
        test_items = items[n_train + n_dev:]

        assert len(train_items) + len(dev_items) + len(test_items) == n

        split_rows["train"].extend(train_items)
        split_rows["dev"].extend(dev_items)
        split_rows["test"].extend(test_items)

        print(
            f"{sentence:30s} "
            f"total={n:4d} | "
            f"train={len(train_items):4d}, "
            f"dev={len(dev_items):4d}, "
            f"test={len(test_items):4d}"
        )

    return split_rows


def write_phoenix_style_corpus(split_name: str, rows):
    out_path = ANNO_ROOT / f"{split_name}.corpus.csv"

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id|folder|signer|annotation"])

        for r in rows:
            annotation = r[LABEL_FIELD]
            line = f"{r['id']}|{r['folder']}|{r['signer']}|{annotation}"
            writer.writerow([line])

    print(f"Saved: {out_path}, samples={len(rows)}, label_field={LABEL_FIELD}")


def has_valid_frame_dir(frame_dir: Path):
    """
    預期結構：
        <split>/<sample_id>/1/*.png
    """
    if not frame_dir.exists():
        return False

    inner_dir = frame_dir / "1"

    if not inner_dir.exists():
        return False

    frame_files = list(inner_dir.glob(f"*{FRAME_EXT}"))

    return len(frame_files) > 0


def analyze_frame_copy_status(split_name: str, rows):
    split_root = FRAME_ROOT / split_name

    complete_ids = []
    missing_or_invalid_ids = []

    for r in rows:
        sample_id = r["id"]
        dst_dir = split_root / sample_id

        if has_valid_frame_dir(dst_dir):
            complete_ids.append(sample_id)
        else:
            missing_or_invalid_ids.append(sample_id)

    return complete_ids, missing_or_invalid_ids


def copy_missing_or_invalid_frame_dirs(split_name: str, rows):
    split_root = FRAME_ROOT / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_valid = 0
    removed_invalid = 0

    for r in tqdm(rows, desc=f"Copying missing {split_name} frames"):
        sample_id = r["id"]

        src_dir = TOTAL_FRAME_ROOT / sample_id
        dst_dir = split_root / sample_id

        if not has_valid_frame_dir(src_dir):
            raise FileNotFoundError(
                f"Missing or invalid source frame directory:\n"
                f"  {src_dir}\n"
                f"Expected structure:\n"
                f"  {src_dir / '1'}/*.png"
            )

        if has_valid_frame_dir(dst_dir):
            skipped_valid += 1
            continue

        if dst_dir.exists():
            shutil.rmtree(dst_dir)
            removed_invalid += 1

        shutil.copytree(src_dir, dst_dir)
        copied += 1

    print(
        f"Frames for {split_name}: "
        f"copied={copied}, "
        f"skipped_valid_existing={skipped_valid}, "
        f"removed_invalid_existing={removed_invalid}, "
        f"target={split_root}"
    )


def recopy_all_frame_dirs(split_name: str, rows):
    split_root = FRAME_ROOT / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    removed_existing = 0

    for r in tqdm(rows, desc=f"Recopying all {split_name} frames"):
        sample_id = r["id"]

        src_dir = TOTAL_FRAME_ROOT / sample_id
        dst_dir = split_root / sample_id

        if not has_valid_frame_dir(src_dir):
            raise FileNotFoundError(
                f"Missing or invalid source frame directory:\n"
                f"  {src_dir}\n"
                f"Expected structure:\n"
                f"  {src_dir / '1'}/*.png"
            )

        if dst_dir.exists():
            shutil.rmtree(dst_dir)
            removed_existing += 1

        shutil.copytree(src_dir, dst_dir)
        copied += 1

    print(
        f"Frames for {split_name}: "
        f"copied={copied}, "
        f"removed_existing={removed_existing}, "
        f"target={split_root}"
    )


def verify_corpus_frame_consistency(split_name: str, rows) -> bool:
    split_root = FRAME_ROOT / split_name
    corpus_ids = {r["id"] for r in rows}

    missing = []
    for r in rows:
        dst_dir = split_root / r["id"]
        if not has_valid_frame_dir(dst_dir):
            missing.append(r["id"])

    orphan = []
    if split_root.exists():
        for d in split_root.iterdir():
            if d.is_dir() and d.name not in corpus_ids:
                orphan.append(d.name)

    ok = True

    if missing:
        ok = False
        print(f"\n[MISMATCH] {split_name}: {len(missing)} sample(s) in corpus but missing/empty in frame dir:")
        for sid in missing[:20]:
            print(f"  {sid}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if orphan:
        ok = False
        print(f"\n[MISMATCH] {split_name}: {len(orphan)} frame dir(s) exist but not in corpus:")
        for sid in sorted(orphan)[:20]:
            print(f"  {sid}")
        if len(orphan) > 20:
            print(f"  ... and {len(orphan) - 20} more")

    if ok:
        print(f"[OK] {split_name}: corpus ({len(corpus_ids)}) matches frame dirs exactly.")

    return ok


def handle_frame_copy_interactively(split_name: str, rows):
    complete_ids, missing_or_invalid_ids = analyze_frame_copy_status(split_name, rows)

    expected = len(rows)
    complete = len(complete_ids)
    missing = len(missing_or_invalid_ids)

    print(
        f"\nFrame status for {split_name}: "
        f"complete={complete}, "
        f"missing_or_invalid={missing}, "
        f"expected={expected}"
    )

    if expected == 0:
        print(f"No samples in {split_name}. Skip frame copying.")
        return

    if missing == 0:
        should_recopy = ask_yes_no(
            f"Frames for {split_name} already look complete. Recopy all {split_name} frames?"
        )

        if should_recopy:
            recopy_all_frame_dirs(split_name, rows)
        else:
            print(f"Skip frame copying for {split_name}. Verifying consistency...")
            ok = verify_corpus_frame_consistency(split_name, rows)
            if not ok:
                raise RuntimeError(
                    f"Consistency check failed for {split_name}. "
                    "Re-run with recopy or fix the mismatched entries manually."
                )
        return

    should_copy_missing = ask_yes_no(
        f"Frames for {split_name} are incomplete. Copy missing or invalid {split_name} frames?"
    )

    if should_copy_missing:
        copy_missing_or_invalid_frame_dirs(split_name, rows)
    else:
        print(f"Skip frame copying for {split_name}. Verifying consistency...")
        ok = verify_corpus_frame_consistency(split_name, rows)
        if not ok:
            raise RuntimeError(
                f"Consistency check failed for {split_name}. "
                "Re-run with recopy or fix the mismatched entries manually."
            )


def print_global_summary(split_rows):
    print("\nGlobal split summary:")

    total = sum(len(v) for v in split_rows.values())

    for split_name in ["train", "dev", "test"]:
        n = len(split_rows[split_name])
        print(f"  {split_name:5s}: {n:4d} ({n / total:.2%})")

    print(f"  total: {total}")


def validate_config():
    if LABEL_FIELD not in {"zh_gloss"}:
        raise ValueError(f"Invalid LABEL_FIELD: {LABEL_FIELD}")

    ratio_sum = sum(SPLIT_RATIO.values())

    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"SPLIT_RATIO must sum to 1.0, got {ratio_sum}")


def main():
    validate_config()

    rows = read_total_corpus(TOTAL_CORPUS)

    print(f"Loaded total samples: {len(rows)}")
    print(f"Split ratio: {SPLIT_RATIO}")
    print(f"Seed: {SEED}")
    print(f"Label field: {LABEL_FIELD}")
    print()

    ids = [r["id"] for r in rows]
    duplicated_ids = [
        sample_id for sample_id, count in Counter(ids).items()
        if count > 1
    ]

    if duplicated_ids:
        raise ValueError(f"Duplicate ids found: {duplicated_ids[:20]}")

    split_rows = stratified_split_by_sentence(
        rows=rows,
        ratio=SPLIT_RATIO,
        seed=SEED,
    )

    print_global_summary(split_rows)
    print()

    for split_name in ["train", "dev", "test"]:
        write_phoenix_style_corpus(split_name, split_rows[split_name])

    print("\nFrame copy check:")
    for split_name in ["train", "dev", "test"]:
        handle_frame_copy_interactively(split_name, split_rows[split_name])

    print("\nDone.")


if __name__ == "__main__":
    main()