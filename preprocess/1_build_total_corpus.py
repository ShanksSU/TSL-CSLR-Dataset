from pathlib import Path
import csv
import re


DATASET_ROOT = Path(r"/root/workspace/Dataset/TSL")
VIDEO_ROOT = DATASET_ROOT / "origin video"
ANNO_ROOT = DATASET_ROOT / "annotations" / "manual"
OUTPUT_CSV = ANNO_ROOT / "total.corpus.csv"


# folder name -> clean English sentence
SENTENCE_MAP = {
    "He has a job": "He has a job",
    "He knows me": "He knows me",
    "I can help you": "I can help you",
    "I can_t hear": "I can't hear",
    "I don_t have a job": "I don't have a job",
    "No smoking here": "No smoking here",
    "Please sign here": "Please sign here",
    "Thank you for helping me": "Thank you for helping me",
    "What_s your name": "What's your name",
    "Where is your home": "Where is your home",
}


# clean English sentence -> Chinese sentence + Chinese gloss
GLOSS_MAP = {
    "He has a job": {
        "zh_sentence": "他有工作",
        "zh_gloss": "他 有 工作",
    },
    "He knows me": {
        "zh_sentence": "他認識我",
        "zh_gloss": "他 認識 我",
    },
    "I can help you": {
        "zh_sentence": "我可以幫助你",
        "zh_gloss": "我 幫助 你 可以",
    },
    "I can't hear": {
        "zh_sentence": "我聽不到",
        "zh_gloss": "我 聽不到",
    },
    "I don't have a job": {
        "zh_sentence": "我沒有工作",
        "zh_gloss": "我 工作 沒有",
    },
    "No smoking here": {
        "zh_sentence": "這裡禁止吸菸",
        "zh_gloss": "這裡 抽菸 不可以",
    },
    "Please sign here": {
        "zh_sentence": "請在這裡簽名",
        "zh_gloss": "請 這裡 簽名",
    },
    "Thank you for helping me": {
        "zh_sentence": "謝謝你幫助我",
        "zh_gloss": "謝謝 你 幫助",
    },
    "What's your name": {
        "zh_sentence": "你叫什麼名字",
        "zh_gloss": "你 名字 什麼",
    },
    "Where is your home": {
        "zh_sentence": "你的家在哪裡",
        "zh_gloss": "你 家 什麼 哪裡",
    },
}


def safe_id(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def parse_video(video_path: Path):
    folder_sentence_raw = video_path.parent.name

    if folder_sentence_raw not in SENTENCE_MAP:
        raise ValueError(f"Unknown sentence folder: {folder_sentence_raw}")

    sentence = SENTENCE_MAP[folder_sentence_raw]

    if sentence not in GLOSS_MAP:
        raise ValueError(f"Missing GLOSS_MAP entry for sentence: {sentence}")

    label_info = GLOSS_MAP[sentence]

    stem = video_path.stem
    parts = stem.rsplit("_", 1)

    if len(parts) != 2:
        raise ValueError(f"Invalid filename format: {video_path.name}")

    prefix, index = parts

    if "_" not in prefix:
        raise ValueError(f"Cannot parse signer from filename: {video_path.name}")

    signer = prefix.split("_", 1)[0]

    sample_id = safe_id(f"{signer}_{sentence}_{index}")
    folder = f"{sample_id}/1/*.png"

    return {
        "id": sample_id,
        "folder": folder,
        "signer": signer,
        "sentence": sentence,
        "zh_sentence": label_info["zh_sentence"],
        "zh_gloss": label_info["zh_gloss"],
        "video_path": str(video_path),
    }


def main():
    if not VIDEO_ROOT.exists():
        raise FileNotFoundError(f"VIDEO_ROOT does not exist: {VIDEO_ROOT}")

    ANNO_ROOT.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(VIDEO_ROOT.glob("*/*.mp4"))

    if not video_paths:
        raise RuntimeError(f"No .mp4 files found under: {VIDEO_ROOT}")

    rows = []
    seen_ids = set()

    for video_path in video_paths:
        row = parse_video(video_path)

        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate sample id found: {row['id']}")

        seen_ids.add(row["id"])
        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "id|folder|signer|sentence|zh_sentence|zh_gloss"
        ])

        for r in rows:
            writer.writerow([
                f"{r['id']}|"
                f"{r['folder']}|"
                f"{r['signer']}|"
                f"{r['sentence']}|"
                f"{r['zh_sentence']}|"
                f"{r['zh_gloss']}"
            ])

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Total videos: {len(rows)}")

    signer_count = {}
    sentence_count = {}

    for r in rows:
        signer_count[r["signer"]] = signer_count.get(r["signer"], 0) + 1
        sentence_count[r["sentence"]] = sentence_count.get(r["sentence"], 0) + 1

    print("\nSigner count:")
    for signer, count in sorted(signer_count.items()):
        print(f"  {signer}: {count}")

    print("\nSentence count:")
    for sentence, count in sorted(sentence_count.items()):
        print(f"  {sentence}: {count}")


if __name__ == "__main__":
    main()