from pathlib import Path
import csv
import re
import shutil
import cv2
from tqdm import tqdm


DATASET_ROOT = Path(r"/root/workspace/Dataset/TSL")

VIDEO_ROOT = DATASET_ROOT / "origin video"
TOTAL_CORPUS = DATASET_ROOT / "annotations" / "manual" / "total.corpus.csv"
FRAME_ROOT = DATASET_ROOT / "features" / "fullFrame-640x480px" / "total"

OUTPUT_SIZE = (640, 480)
IMAGE_EXT = ".png"


def ask_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False
        print("Invalid input. Please enter 'y' or 'n'.")


def safe_id(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def read_total_corpus(corpus_path: Path):
    if not corpus_path.exists():
        raise FileNotFoundError(f"Cannot find corpus file: {corpus_path}")

    corpus = {}

    with open(corpus_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        header = next(reader)
        if len(header) != 1:
            raise ValueError(
                "Expected Phoenix-style single-column CSV, "
                f"but got header: {header}"
            )

        expected_header = "id|folder|signer|sentence|zh_sentence|zh_gloss"
        if header[0] != expected_header:
            raise ValueError(
                f"Unexpected header:\n"
                f"  got     : {header[0]}\n"
                f"  expected: {expected_header}"
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

            if sample_id in corpus:
                raise ValueError(f"Duplicate sample id in corpus: {sample_id}")

            corpus[sample_id] = {
                "folder": folder,
                "signer": signer,
                "sentence": sentence,
                "zh_sentence": zh_sentence,
                "zh_gloss": zh_gloss,
            }

    return corpus


def build_video_index(video_root: Path):
    if not video_root.exists():
        raise FileNotFoundError(f"VIDEO_ROOT does not exist: {video_root}")

    video_paths = sorted(video_root.glob("*/*.mp4"))

    if not video_paths:
        raise RuntimeError(f"No .mp4 files found under: {video_root}")

    video_index = {}

    for video_path in video_paths:
        sample_id = safe_id(video_path.stem)

        if sample_id in video_index:
            raise ValueError(
                f"Duplicate video id detected:\n"
                f"  id: {sample_id}\n"
                f"  video 1: {video_index[sample_id]}\n"
                f"  video 2: {video_path}"
            )

        video_index[sample_id] = video_path

    return video_index


def extract_frames(video_path: Path, output_dir: Path, overwrite: bool):
    if output_dir.exists():
        existing_frames = sorted(output_dir.glob(f"*{IMAGE_EXT}"))

        if existing_frames and not overwrite:
            return {
                "status": "skipped_exists",
                "num_frames": len(existing_frames),
                "orig_width": None,
                "orig_height": None,
                "fps": None,
            }

        if overwrite:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 1
    valid_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame.shape[1] != OUTPUT_SIZE[0] or frame.shape[0] != OUTPUT_SIZE[1]:
            frame = cv2.resize(
                frame,
                OUTPUT_SIZE,
                interpolation=cv2.INTER_LANCZOS4,
            )

        frame_path = output_dir / f"{frame_idx:06d}{IMAGE_EXT}"

        ok = cv2.imwrite(str(frame_path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write frame: {frame_path}")

        frame_idx += 1
        valid_count += 1

    cap.release()

    if valid_count == 0:
        raise RuntimeError(f"No frame extracted from: {video_path}")

    return {
        "status": "extracted",
        "num_frames": valid_count,
        "orig_width": orig_width,
        "orig_height": orig_height,
        "fps": fps,
    }


def check_existing_frames(corpus: dict) -> tuple[int, int]:
    complete, missing = 0, 0
    for sample_id, item in corpus.items():
        folder_pattern = item["folder"]
        output_dir = FRAME_ROOT / Path(folder_pattern).parent
        existing = list(output_dir.glob(f"*{IMAGE_EXT}")) if output_dir.exists() else []
        if existing:
            complete += 1
        else:
            missing += 1
    return complete, missing


def main():
    FRAME_ROOT.mkdir(parents=True, exist_ok=True)

    corpus = read_total_corpus(TOTAL_CORPUS)
    video_index = build_video_index(VIDEO_ROOT)
    complete, missing = check_existing_frames(corpus)
    print(f"\nFrame extraction status: complete={complete}, missing={missing}, total={len(corpus)}")

    if missing == 0:
        overwrite = ask_yes_no(
            "All frames already exist. Re-extract and overwrite all frames?"
        )
    else:
        overwrite = ask_yes_no(
            f"{missing} sample(s) have no frames yet. "
            f"Also overwrite the {complete} already-existing sample(s)?"
        )

    if overwrite:
        print("Mode: OVERWRITE — all existing frames will be deleted and re-extracted.")
    else:
        print("Mode: SKIP existing — only missing samples will be extracted.")

    report_rows = []
    missing_video_ids = []
    extra_video_ids = sorted(set(video_index.keys()) - set(corpus.keys()))

    for sample_id, item in tqdm(corpus.items(), desc="Extracting frames"):
        if sample_id not in video_index:
            missing_video_ids.append(sample_id)
            continue

        video_path = video_index[sample_id]
        folder_pattern = item["folder"]
        output_dir = FRAME_ROOT / Path(folder_pattern).parent

        result = extract_frames(video_path, output_dir, overwrite=overwrite)

        report_rows.append({
            "id": sample_id,
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "status": result["status"],
            "num_frames": result["num_frames"],
            "orig_width": result["orig_width"],
            "orig_height": result["orig_height"],
            "fps": result["fps"],
            "signer": item["signer"],
            "sentence": item["sentence"],
            "zh_sentence": item["zh_sentence"],
            "zh_gloss": item["zh_gloss"],
        })

    if missing_video_ids:
        print("\n[ERROR] These ids exist in total.corpus.csv but no matching mp4 was found:")
        for sample_id in missing_video_ids[:50]:
            print(f"  {sample_id}")
        if len(missing_video_ids) > 50:
            print(f"  ... and {len(missing_video_ids) - 50} more")
        raise RuntimeError(
            f"Missing {len(missing_video_ids)} videos. "
            "Fix id generation or video filenames before continuing."
        )

    if extra_video_ids:
        print("\n[Warning] These mp4 files exist but are not listed in total.corpus.csv:")
        for sample_id in extra_video_ids[:50]:
            print(f"  {sample_id}")
        if len(extra_video_ids) > 50:
            print(f"  ... and {len(extra_video_ids) - 50} more")

    report_path = DATASET_ROOT / "annotations" / "manual" / "frame_extraction_total_report.csv"

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id", "video_path", "output_dir", "status",
            "num_frames", "orig_width", "orig_height", "fps",
            "signer", "sentence", "zh_sentence", "zh_gloss",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nDone.")
    print(f"Corpus samples : {len(corpus)}")
    print(f"Videos indexed : {len(video_index)}")
    print(f"Videos processed: {len(report_rows)}")
    print(f"Frame root     : {FRAME_ROOT}")
    print(f"Report saved   : {report_path}")


if __name__ == "__main__":
    main()