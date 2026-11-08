"""Core video analysis and unique frame extraction pipeline."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterator, List, Optional, Tuple
import warnings

import cv2
import imagehash
import numpy as np
from PIL import Image
from scenedetect import ContentDetector, FrameTimecode
import torch
import torch.nn.functional as F
from tqdm import tqdm

from contact_sheet import generate_contact_sheet

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


@dataclass
class SavedFrameState:
    """Lightweight representation of a saved frame for similarity comparison."""

    index: int
    timestamp: float
    scene_id: int
    cls_feat: np.ndarray       # Normalized (384,)
    patch_feat: np.ndarray     # Normalized (N_patches, 384)
    phash: imagehash.ImageHash
    dhash: imagehash.ImageHash
    thumb_rgb: np.ndarray      # (64, 64, 3) uint8 thumbnail
    file_path: Path


@dataclass
class VideoMetadata:
    """Basic metadata for an input video stream."""

    width: int
    height: int
    fps: float
    duration: float
    codec_name: str


def resolve_device(requested_device: str = "auto") -> torch.device:
    """Resolve the compute device ('auto', 'cuda', or 'cpu') with safe fallback."""
    req = requested_device.strip().lower()
    if req == "cpu":
        return torch.device("cpu")
    if req in ("auto", "cuda"):
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except Exception as exc:
                print(f"[WARNING] CUDA initialization failed ({exc}); falling back to CPU.")
                return torch.device("cpu")
        if req == "cuda":
            print("[WARNING] CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option '{requested_device}'. Choose auto, cuda, or cpu.")


def check_ffmpeg_available() -> bool:
    """Check whether the ffmpeg binary is available on PATH."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def probe_video_metadata(video_path: Path) -> Optional[VideoMetadata]:
    """Probe video dimensions, FPS, duration, and codec using ffprobe or OpenCV fallback."""
    if shutil.which("ffprobe") is not None:
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,duration:format=duration",
                "-of",
                "json",
                str(video_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(res.stdout)
            streams = info.get("streams", [])
            if streams:
                st = streams[0]
                width = int(st.get("width", 0))
                height = int(st.get("height", 0))
                codec_name = str(st.get("codec_name", "unknown"))

                fps = 30.0
                fps_str = st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1"
                if "/" in fps_str:
                    num, den = fps_str.split("/", 1)
                    if float(den) > 0 and float(num) > 0:
                        fps = float(num) / float(den)
                elif float(fps_str) > 0:
                    fps = float(fps_str)

                duration = 0.0
                if st.get("duration") is not None:
                    duration = float(st["duration"])
                elif info.get("format", {}).get("duration") is not None:
                    duration = float(info["format"]["duration"])

                if width > 0 and height > 0:
                    return VideoMetadata(
                        width=width,
                        height=height,
                        fps=fps,
                        duration=duration,
                        codec_name=codec_name,
                    )
        except Exception:
            pass

    # Fallback to OpenCV metadata probing
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    duration = total_frames / fps if total_frames > 0 else 0.0
    if width <= 0 or height <= 0:
        return None
    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        codec_name="opencv",
    )


def stream_sampled_frames(
    video_path: Path,
    meta: VideoMetadata,
    sample_fps: float,
) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (timestamp_sec, rgb_frame) at sample_fps using FFmpeg pipe with OpenCV fallback."""
    frame_bytes = meta.width * meta.height * 3

    if check_ffmpeg_available():
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={sample_fps:.4f}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=frame_bytes * 2,
        )
        sample_idx = 0
        try:
            assert proc.stdout is not None
            while True:
                raw = proc.stdout.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    break
                rgb_frame = np.frombuffer(raw, dtype=np.uint8).reshape((meta.height, meta.width, 3)).copy()
                timestamp = sample_idx / sample_fps
                yield timestamp, rgb_frame
                sample_idx += 1
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            proc.wait()

        if sample_idx > 0:
            return

    # Fallback to OpenCV VideoCapture if FFmpeg is unavailable
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    sample_interval_sec = 1.0 / max(0.1, float(sample_fps))
    next_sample_sec = 0.0
    frame_idx = 0
    try:
        while True:
            ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None:
                break
            curr_sec = frame_idx / meta.fps
            if curr_sec + 1e-6 >= next_sample_sec:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                yield curr_sec, rgb_frame
                next_sample_sec += sample_interval_sec
            frame_idx += 1
    finally:
        cap.release()


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH-MM-SS.mmm for frame filenames."""
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}.{millis:03d}"


def compute_dino_input_dims(width: int, height: int, max_side: int = 336, patch_size: int = 14) -> Tuple[int, int]:
    """Compute aspect-ratio-preserving dimensions divisible by 14 for DINOv2 ViT-S/14."""
    scale = max_side / max(width, height)
    target_w = max(patch_size * 4, int(round((width * scale) / patch_size)) * patch_size)
    target_h = max(patch_size * 4, int(round((height * scale) / patch_size)) * patch_size)
    return target_w, target_h


class FeatureExtractor:
    """DINOv2 ViT-S/14 visual feature extractor with CUDA/CPU support."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        cache_dir = Path(__file__).resolve().parent / ".cache" / "torch"
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_dir))
        print(f"[INFO] Loading DINOv2 ViT-S/14 model on {self.device}...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=True,
                verbose=False,
            )
        self.model.eval()
        self.model.to(self.device)

        # ImageNet normalization constants expected by DINOv2
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def extract_batch(
        self,
        resized_rgb_frames: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract normalized CLS and spatial patch features for a batch of resized RGB frames.

        Args:
            resized_rgb_frames: List of HxWx3 uint8 RGB numpy arrays already resized to DINOv2 dims.

        Returns:
            Tuple of:
                - cls_features: (B, 384) float32 numpy array
                - patch_features: (B, N_patches, 384) float32 numpy array
        """
        batch_np = np.stack(resized_rgb_frames, axis=0)  # (B, H, W, 3), uint8
        tensor = torch.from_numpy(batch_np).to(self.device, non_blocking=True)
        tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
        tensor = (tensor - self.mean) / self.std

        features = self.model.forward_features(tensor)
        cls_tokens = F.normalize(features["x_norm_clstoken"], p=2, dim=-1)
        patch_tokens = F.normalize(features["x_norm_patchtokens"], p=2, dim=-1)

        return (
            cls_tokens.cpu().numpy().astype(np.float32),
            patch_tokens.cpu().numpy().astype(np.float32),
        )


def compute_visual_distance(
    cls_feat: np.ndarray,
    patch_feat: np.ndarray,
    ref_cls: np.ndarray,
    ref_patch: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute global, 65th-percentile spatial patch, and combined visual distances.

    Returns:
        Tuple of (combined_score, cls_distance, p65_patch_distance)
    """
    cls_sim = float(np.dot(cls_feat, ref_cls))
    cls_dist = max(0.0, 1.0 - cls_sim)

    # Patch-wise cosine distance across spatial tokens
    patch_sims = np.sum(patch_feat * ref_patch, axis=-1)
    patch_dists = np.clip(1.0 - patch_sims, 0.0, 2.0)

    # 65th-percentile spatial patch distance captures animal pose/framing changes
    # while ignoring small localized subtitle pop-ins (~15-20% of patches)
    p65_patch_dist = float(np.percentile(patch_dists, 65))
    mean_patch_dist = float(np.mean(patch_dists))

    combined = 0.55 * cls_dist + 0.25 * mean_patch_dist + 0.20 * p65_patch_dist
    return combined, cls_dist, p65_patch_dist


def is_near_duplicate(
    cls_feat: np.ndarray,
    patch_feat: np.ndarray,
    phash: imagehash.ImageHash,
    dhash: imagehash.ImageHash,
    thumb_rgb: np.ndarray,
    recent_saved: List[SavedFrameState],
    sensitivity: float,
) -> bool:
    """Check if candidate frame is nearly identical to any recently saved frame."""
    if not recent_saved:
        return False

    # Higher sensitivity allows slightly closer hashes if visual features differ meaningfully
    max_dup_hash_dist = max(4, int(round(12 * (1.0 - 0.55 * sensitivity))))
    min_visual_diff = 0.04 + 0.08 * (1.0 - sensitivity)

    for prev in reversed(recent_saved[-6:]):
        phash_dist = phash - prev.phash
        dhash_dist = dhash - prev.dhash
        combined_dist, cls_dist, p65_dist = compute_visual_distance(
            cls_feat, patch_feat, prev.cls_feat, prev.patch_feat
        )

        # Direct thumbnail pixel difference (immune to flat-background dhash noise)
        pixel_diffs = np.abs(thumb_rgb.astype(np.float32) - prev.thumb_rgb.astype(np.float32))
        mean_pixel_diff = float(np.mean(pixel_diffs))
        p95_pixel_diff = float(np.percentile(pixel_diffs, 95))

        if mean_pixel_diff < 4.5 and p95_pixel_diff < 15.0 and cls_dist < 0.06:
            return True

        # Identical or near-identical perceptual hash AND small DINOv2 difference
        if min(phash_dist, dhash_dist) <= max_dup_hash_dist:
            if combined_dist < min_visual_diff * 1.3 and p65_dist < min_visual_diff * 1.5:
                return True

        # Extremely close DINOv2 representation (e.g. minor compression or subtitle flicker)
        if combined_dist < min_visual_diff * 0.85 and p65_dist < min_visual_diff:
            return True

    return False


def process_video(
    video_path: Path,
    output_root: Path,
    extractor: FeatureExtractor,
    sensitivity: float = 0.8,
    sample_fps: float = 5.0,
    batch_size: int = 16,
    jpeg_quality: int = 95,
) -> Optional[Path]:
    """Analyze a single video and extract visually unique frames + contact sheet.

    Args:
        video_path: Path to the input video file.
        output_root: Base output directory.
        extractor: Initialized DINOv2 FeatureExtractor.
        sensitivity: Extraction sensitivity in [0.0, 1.0] (higher = more frames extracted).
        sample_fps: Number of frames sampled per second of video.
        batch_size: Batch size for DINOv2 inference.
        jpeg_quality: JPEG save quality (1-100).

    Returns:
        Path to the video's output directory, or None if extraction failed.
    """
    if not video_path.is_file():
        print(f"[ERROR] Video file not found: {video_path}")
        return None

    meta = probe_video_metadata(video_path)
    if meta is None:
        print(f"[ERROR] Failed to read video metadata from: {video_path}")
        return None

    print(
        f"\n[INFO] Processing '{video_path.name}' "
        f"({meta.width}x{meta.height}, codec={meta.codec_name}, {meta.fps:.2f} FPS, ~{meta.duration:.1f}s)"
    )

    # Prepare output directories
    video_out_dir = output_root / video_path.stem
    frames_dir = video_out_dir / "frames"
    if frames_dir.exists():
        for existing_jpg in frames_dir.glob("*.jpg"):
            existing_jpg.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Calculate change threshold from user sensitivity (clamped to [0.05, 0.98])
    sens = float(np.clip(sensitivity, 0.05, 0.98))
    change_threshold = 0.06 + 0.45 * ((1.0 - sens) ** 1.1)

    dino_w, dino_h = compute_dino_input_dims(meta.width, meta.height)

    # Online PySceneDetect ContentDetector on sampled frames
    scene_detector = ContentDetector(threshold=24.0, min_scene_len=1)
    current_scene_id = 0

    saved_states: List[SavedFrameState] = []
    batch_full_rgb: List[np.ndarray] = []
    batch_resized_rgb: List[np.ndarray] = []
    batch_timestamps: List[float] = []
    batch_scene_ids: List[int] = []

    def evaluate_and_flush_batch() -> None:
        if not batch_full_rgb:
            return

        cls_feats, patch_feats = extractor.extract_batch(batch_resized_rgb)

        for idx_in_batch, rgb_img in enumerate(batch_full_rgb):
            ts = batch_timestamps[idx_in_batch]
            scene_id = batch_scene_ids[idx_in_batch]
            cls_f = cls_feats[idx_in_batch]
            patch_f = patch_feats[idx_in_batch]

            # Compute perceptual hashes on a 64x64 thumbnail from the already-downscaled frame
            small_rgb = batch_resized_rgb[idx_in_batch]
            thumb_rgb = cv2.resize(small_rgb, (64, 64), interpolation=cv2.INTER_AREA)
            hash_thumb = Image.fromarray(thumb_rgb)
            curr_phash = imagehash.phash(hash_thumb)
            curr_dhash = imagehash.dhash(hash_thumb)

            should_save = False
            if not saved_states:
                should_save = True
            else:
                if is_near_duplicate(
                    cls_f, patch_f, curr_phash, curr_dhash, thumb_rgb, saved_states, sens
                ):
                    should_save = False
                else:
                    last_saved = saved_states[-1]
                    combined_dist, _, _ = compute_visual_distance(
                        cls_f, patch_f, last_saved.cls_feat, last_saved.patch_feat
                    )
                    # Lower threshold across a detected scene cut so new angles/shots are captured
                    effective_thresh = (
                        change_threshold * 0.55
                        if scene_id != last_saved.scene_id
                        else change_threshold
                    )
                    if combined_dist >= effective_thresh:
                        should_save = True

            if should_save:
                seq_num = len(saved_states) + 1
                ts_str = format_timestamp(ts)
                frame_filename = f"{seq_num:03d}_{ts_str}.jpg"
                frame_path = frames_dir / frame_filename
                pil_img = Image.fromarray(rgb_img)
                pil_img.save(frame_path, format="JPEG", quality=jpeg_quality)

                saved_states.append(
                    SavedFrameState(
                        index=seq_num,
                        timestamp=ts,
                        scene_id=scene_id,
                        cls_feat=cls_f,
                        patch_feat=patch_f,
                        phash=curr_phash,
                        dhash=curr_dhash,
                        thumb_rgb=thumb_rgb,
                        file_path=frame_path,
                    )
                )

        batch_full_rgb.clear()
        batch_resized_rgb.clear()
        batch_timestamps.clear()
        batch_scene_ids.clear()

    expected_samples = (
        max(1, int(round(meta.duration * sample_fps))) if meta.duration > 0 else None
    )

    sample_idx = 0
    with tqdm(
        total=expected_samples,
        desc=f"Analyzing {video_path.name}",
        unit="sample",
    ) as pbar:
        for curr_sec, rgb_frame in stream_sampled_frames(video_path, meta, sample_fps):
            resized_rgb = cv2.resize(rgb_frame, (dino_w, dino_h), interpolation=cv2.INTER_AREA)
            resized_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)

            # Detect scene cut via PySceneDetect ContentDetector
            cuts = scene_detector.process_frame(FrameTimecode(sample_idx, sample_fps), resized_bgr)
            if cuts:
                current_scene_id += 1

            batch_full_rgb.append(rgb_frame)
            batch_resized_rgb.append(resized_rgb)
            batch_timestamps.append(curr_sec)
            batch_scene_ids.append(current_scene_id)

            if len(batch_full_rgb) >= batch_size:
                evaluate_and_flush_batch()

            sample_idx += 1
            pbar.update(1)

    evaluate_and_flush_batch()

    if not saved_states:
        print(f"[WARNING] No frames were extracted from {video_path.name}.")
        return None

    print(f"[INFO] Detected {current_scene_id + 1} scene segment(s) across {sample_idx} sampled frames.")

    # Generate contact sheet
    contact_sheet_path = video_out_dir / "contact_sheet.jpg"
    frame_entries = [(st.file_path, st.timestamp) for st in saved_states]
    generate_contact_sheet(frame_entries, contact_sheet_path)

    print(
        f"[SUCCESS] Extracted {len(saved_states)} unique frame(s) to '{frames_dir}' "
        f"and saved contact sheet to '{contact_sheet_path}'."
    )
    return video_out_dir


def collect_video_files(input_path: Path) -> List[Path]:
    """Return a sorted list of supported video files from a file or directory path."""
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        raise ValueError(
            f"Unsupported file extension '{input_path.suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if input_path.is_dir():
        videos = [
            p for p in sorted(input_path.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return videos

    raise FileNotFoundError(f"Input path does not exist: {input_path}")
