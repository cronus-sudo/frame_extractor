# Cronus Wildlife Reel Frame Extractor

A lightweight Python CLI tool that analyzes animal fact explainer videos and extracts **visually unique frames of animals** (scene cuts, camera angle changes, close-ups/wide shots, and in-shot animal pose or movement changes) while filtering out near-duplicate frames.

## Features

- **Multi-format & Multi-aspect Support:** Supports `.mp4`, `.mov`, `.mkv`, and `.webm` in vertical (9:16), horizontal (16:9), or square (1:1) aspect ratios.
- **Scene Cut Detection:** Uses `PySceneDetect` to detect hard cuts and transitions.
- **Continuous-Shot Pose & Composition Detection:** Uses **DINOv2 ViT-S/14** (`facebookresearch/dinov2`) global (`CLS`) and spatial patch embeddings to identify meaningful changes in animal pose, head turns, actions, or framing within a continuous shot.
- **Near-Duplicate Removal:** Combines `ImageHash` (`phash` + `dhash`) perceptual hashing and DINOv2 feature similarity to avoid saving bursts of nearly identical frames.
- **Contact Sheet Generation:** Automatically builds a labeled `contact_sheet.jpg` alongside the full-resolution JPEG frames for rapid visual review.
- **Hardware Acceleration:** Uses NVIDIA CUDA automatically when available, with seamless CPU fallback.

---

## 1. Ubuntu (WSL2) Installation Instructions

Open your Ubuntu WSL2 terminal and install FFmpeg and Python virtual environment tools:

```bash
sudo apt update
sudo apt install -y ffmpeg python3 python3-venv python3-pip
```

Navigate to the project directory, create/activate the virtual environment, and install dependencies:

```bash
cd /home/ghost/workspace/frame_extractor
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Running the Tool

### Analyze a Single Video

```bash
python main.py \
  --input ~/videos/wildlife.mp4 \
  --output ~/cronus/frames \
  --sensitivity 0.8 \
  --sample-fps 5
```

### Analyze a Directory of Videos

```bash
python main.py \
  --input ~/videos/reels/ \
  --output ./output \
  --sensitivity 0.8 \
  --sample-fps 5 \
  --device auto
```

### CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--input`, `-i` | *(Required)* | Input video file (`.mp4`, `.mov`, `.mkv`, `.webm`) or folder containing videos. |
| `--output`, `-o` | `output` | Destination root directory for extracted frames and contact sheets. |
| `--sensitivity`, `-s` | `0.8` | Extraction sensitivity in `(0.0, 1.0]`. Higher values capture subtler animal pose/framing changes; lower values require larger visual changes. |
| `--sample-fps`, `-f` | `5.0` | Number of frames sampled and analyzed per second of video. |
| `--device`, `-d` | `auto` | Compute device for DINOv2 inference: `auto`, `cuda`, or `cpu`. |

---

## 3. Output Structure

For each processed video, a dedicated folder is created inside `--output`:

```text
output/
└── wildlife_reel/
    ├── frames/
    │   ├── 001_00-00-00.000.jpg
    │   ├── 002_00-00-01.200.jpg
    │   ├── 003_00-00-02.600.jpg
    │   └── ...
    └── contact_sheet.jpg
```

- **`frames/`**: Extracted frames saved at the video's original resolution and aspect ratio in chronological order with sequential index and `HH-MM-SS.mmm` timestamp.
- **`contact_sheet.jpg`**: Grid overview of all extracted frames with their filenames/timestamps.

# frame_extractor
# frame_extractor
