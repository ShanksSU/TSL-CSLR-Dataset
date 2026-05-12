# TSL Dataset

## Introduction

TSL (Taiwan Sign Language) is a continuous sign language recognition dataset collected for research purposes. It contains video recordings of 10 common Taiwanese Sign Language sentences performed by 5 signers, with sentence-level annotations for training continuous sign language recognition models.

The dataset follows the [Phoenix2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) pipeline convention and is compatible with CTC-based continuous sign language recognition frameworks.

---

## Signers

| Signer | Samples |
|--------|---------|
| Leo    | 351     |
| Alex   | 350     |
| Auther | 350     |
| Jerry  | 349     |
| David  | 346     |
| **Total** | **1746** |

---

## Sentence Classes

| Sentence | Chinese | TSL Gloss | Total | Train | Dev | Test |
|----------|---------|-----------|------:|------:|----:|-----:|
| He has a job | 他有工作。 | 他 工作 有 | 175 | 105 | 35 | 35 |
| He knows me | 他認識我。 | 他 我 認識 | 176 | 106 | 35 | 35 |
| I can help you | 我可以幫你。 | 我 幫忙 可以 | 175 | 105 | 35 | 35 |
| I can't hear | 我聽不見。 | 我 聽不見 | 175 | 105 | 35 | 35 |
| I don't have a job | 我沒有工作。 | 我 工作 沒有 | 175 | 105 | 35 | 35 |
| No smoking here | 這裡禁止吸菸。 | 這 抽煙 不可以 | 175 | 105 | 35 | 35 |
| Please sign here | 請在這裡簽名。 | 請 這 簽名 | 174 | 104 | 35 | 35 |
| Thank you for helping me | 謝謝你幫助我。 | 謝謝 你 幫忙 | 174 | 104 | 35 | 35 |
| What's your name | 你叫什麼名字？ | 你 名字 什麼 | 174 | 104 | 35 | 35 |
| Where is your home | 你家在哪裡？ | 你 家 什麼 哪裡 | 173 | 104 | 35 | 34 |

---

## Split Summary

| Split | Samples | Ratio |
|-------|--------:|------:|
| Train | 1047 | 59.97% |
| Dev   | 350  | 20.05% |
| Test  | 349  | 19.99% |
| **Total** | **1746** | |

Splits are stratified by sentence class with `SEED=0` to ensure reproducibility.

---

## Environment Setup

### Docker (Recommended)

Build the image:

```bash
docker build -t <image_name> .
```

Run the container:

```bash
docker run -d --gpus all -p 8025:22 --shm-size=16g \
  -v "C:\Users\<username>\Downloads\Dataset:/root/workspace/Dataset" \
  --name <container_name> <image_name>
```

The container runs an SSH server on port 8025. Default root password is `123456`.

Environment inside container:
- CUDA 12.8.1 + cuDNN
- Python 3.10 (conda env `torch_222`)
- PyTorch 2.2.2 + torchvision 0.17.2 (cu121)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch==2.2.2`
- `opencv-python`
- `numpy`
- `einops`
- `tqdm`

### Evaluation Tool (Optional)

[sclite](https://github.com/kaldi-asr/kaldi) provides detailed WER statistics. After installing Kaldi, create a soft link:

```bash
ln -s PATH_TO_KALDI/tools/sctk-20159b5/bin/sclite ./software/sclite
```

A Python-based WER evaluation tool is also provided under `evaluation/slr_eval/` for convenience.

---

## Preprocessing Pipeline

### Step 1 — Build total corpus

Scans all `.mp4` files under `origin video/` and generates `total.corpus.csv`.

```bash
python 1_build_total_corpus.py
```

### Step 2 — Extract frames

Extracts frames from each video into `features/fullFrame-640x480px/total/`.

```bash
python 2_extract_total_frames.py
```

Set `OVERWRITE = True` in the script to force re-extraction of existing frames.

### Step 3 — Split corpus and copy frames

Performs stratified split by sentence class and copies frames into `train/dev/test/` directories. Generates Phoenix-style `.corpus.csv` for each split.

```bash
python 3_split_total_corpus.py
```

The script will interactively ask whether to copy or skip frames for each split. If skipping, a consistency check is automatically run to verify that corpus entries match the frame directories.

### Step 4 — Generate info and gloss dictionary

Generates `*_info.npy` and `gloss_dict.npy` used by the training framework. Optionally resizes frames from 640×480 to 256×256.

```bash
python dataset_preprocess-TSL.py --process-image --multiprocessing
```

---

## Annotation Format

### `*.corpus.csv`

Phoenix-style pipe-delimited single-column CSV:

```
id|folder|signer|annotation
Alex_He_has_a_job_001|Alex_He_has_a_job_001/1/*.png|Alex|他 有 工作
```

### `total.corpus.csv`

Extended format with full metadata:

```
id|folder|signer|sentence|zh_sentence|zh_gloss
Alex_He_has_a_job_001|Alex_He_has_a_job_001/1/*.png|Alex|He has a job|他有工作|他 有 工作
```

### `gloss_dict.npy`

Python dict mapping each gloss token to `[index, frequency]`:

```python
gloss_dict = np.load("TSL/gloss_dict.npy", allow_pickle=True).item()
# e.g. {"他": [1, 350], "有": [2, 175], ...}
```

---

## Training

```bash
python main.py --work-dir PATH_TO_SAVE_RESULTS --config configs/baseline.yaml --device 0
```

Configuration priority: command line > config file > argparse defaults.

## Inference

```bash
python main.py --config ./configs/baseline.yaml --device 0 --load-weights path_to_weight.pt --phase test
```

---

## Ground Truth STM

Each split generates a NIST SCLITE-compatible `.stm` file for evaluation:

```
TSL/TSL-groundtruth-train.stm
TSL/TSL-groundtruth-dev.stm
TSL/TSL-groundtruth-test.stm
```

Format per line:
```
<fileid> 1 <signer> 0.0 1.79769e+308 <gloss sequence>
```

---

## Citation

The raw video data was obtained from the authors of the following paper. The original dataset contains only videos and their corresponding English sentences. This repository extends it with gloss annotations, train/dev/test splits, and frame extraction.

**If you wish to access the dataset, please contact the original author Huang, Wei-Hao or the maintainer of this repository.**

**If you use this dataset in your work, you must cite the following paper.**

```bibtex
@InProceedings{huang2025tsl,
    author    = {Huang, Wei-Hao and Zhao, Qiangfu},
    title     = {Taiwan Sign Language Recognition Based on MediaPipe and Deep Learning},
    booktitle = {2025 International Conference on Advanced Robotics and Intelligent Systems (ARIS)},
    year      = {2025},
    pages     = {1--6},
    doi       = {10.1109/ARIS66143.2025.11163460}
}
```

---

## Acknowledgement

The baseline model is based on [VAC (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html). The raw video dataset was provided by Huang, Wei-Hao. Many thanks for their great work!

```bibtex
@InProceedings{Min_2021_ICCV,
    author    = {Min, Yuecong and Hao, Aiming and Chai, Xiujuan and Chen, Xilin},
    title     = {Visual Alignment Constraint for Continuous Sign Language Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {11542-11551}
}

@InProceedings{huang2025tsl,
    author    = {Huang, Wei-Hao and Zhao, Qiangfu},
    title     = {Taiwan Sign Language Recognition Based on MediaPipe and Deep Learning},
    booktitle = {2025 International Conference on Advanced Robotics and Intelligent Systems (ARIS)},
    year      = {2025},
    pages     = {1--6},
    doi       = {10.1109/ARIS66143.2025.11163460}
}
```