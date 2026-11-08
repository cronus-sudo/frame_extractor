#!/usr/bin/env python3
"""CLI entry point for the Cronus Wildlife Reel Frame Extractor."""

import argparse
from pathlib import Path
import sys

from extractor import (
    FeatureExtractor,
    check_ffmpeg_available,
    collect_video_files,
    process_video,
    resolve_device,
)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Cronus Wildlife Reel Frame Extractor: Analyze animal explainer videos "
            "and extract visually unique frames and a contact sheet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to a single video file (.mp4, .mov, .mkv, .webm) or a directory of videos.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output",
        help="Destination directory for extracted frames and contact sheets.",
    )
    parser.add_argument(
        "--sensitivity",
        "-s",
        type=float,
        default=0.8,
        help="Extraction sensitivity between 0.0 and 1.0 (higher extracts more subtle pose/angle changes).",
    )
    parser.add_argument(
        "--sample-fps",
        "-f",
        type=float,
        default=5.0,
        help="Number of frames analyzed per second of video.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device for DINOv2 model inference ('auto', 'cuda', or 'cpu').",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Main CLI execution flow."""
    args = parse_args(argv)

    if not (0.0 < args.sensitivity <= 1.0):
        print("[ERROR] --sensitivity must be in the range (0.0, 1.0].", file=sys.stderr)
        return 1

    if args.sample_fps <= 0.0:
        print("[ERROR] --sample-fps must be greater than 0.", file=sys.stderr)
        return 1

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not check_ffmpeg_available():
        print(
            "[WARNING] FFmpeg binary not detected on PATH. OpenCV built-in decoding will be used, "
            "but installing ffmpeg (`sudo apt install ffmpeg`) is recommended."
        )

    try:
        video_files = collect_video_files(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if not video_files:
        print(f"[ERROR] No supported video files (.mp4, .mov, .mkv, .webm) found in: {input_path}", file=sys.stderr)
        return 1

    print(f"[INFO] Found {len(video_files)} video(s) to process.")
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        device = resolve_device(args.device)
        extractor = FeatureExtractor(device=device)
    except Exception as exc:
        print(f"[ERROR] Failed to initialize DINOv2 feature extractor: {exc}", file=sys.stderr)
        return 1

    success_count = 0
    for video_path in video_files:
        try:
            res = process_video(
                video_path=video_path,
                output_root=output_root,
                extractor=extractor,
                sensitivity=args.sensitivity,
                sample_fps=args.sample_fps,
            )
            if res is not None:
                success_count += 1
        except Exception as exc:
            print(f"[ERROR] Unexpected error while processing '{video_path.name}': {exc}", file=sys.stderr)

    print(f"\n[DONE] Successfully processed {success_count}/{len(video_files)} video(s).")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

