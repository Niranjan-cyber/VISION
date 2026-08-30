"""
Command-line script to extract raw or enhanced frames from CCTV surveillance videos.
Usage:
    # Extract raw frames
    python scripts/extract_frames.py --video videos/VIRAT_S_010205_04_000545_000576.mp4 --step 5

    # Extract and ENHANCE frames directly
    python scripts/extract_frames.py --video videos/VIRAT_S_010205_04_000545_000576.mp4 --enhance --step 5
"""

import os
import argparse
import sys
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.video.frame_extractor import FrameExtractor
from src.video.reader import VideoReader
from src.enhancement.manager import EnhancementManager


def main():
    parser = argparse.ArgumentParser(description="SIH 2026 CCTV Frame Extractor")
    parser.add_argument("--video", type=str, default=None, help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, default="videos", help="Path to folder containing videos")
    parser.add_argument("--output_dir", type=str, default="frames", help="Output directory for frames")
    parser.add_argument("--enhance", action="store_true", help="Apply AI enhancement pipeline to extracted frames")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "zero_dce", "realesrgan", "rvrt", "all"], help="Enhancement mode if --enhance is used")
    parser.add_argument("--step", type=int, default=1, help="Extract every N-th frame (default: 1)")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to extract per video")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg"], help="Image format")

    args = parser.parse_args()

    if args.enhance:
        manager = EnhancementManager()
        videos_to_process = [args.video] if args.video else [
            os.path.join(args.video_dir, f) for f in sorted(os.listdir(args.video_dir))
            if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]

        for vpath in videos_to_process:
            vname = os.path.splitext(os.path.basename(vpath))[0]
            out_folder = os.path.join(args.output_dir, "enhanced", vname)
            os.makedirs(out_folder, exist_ok=True)
            print(f"\n[Extract & Enhance] Processing: {os.path.basename(vpath)} -> {out_folder}")

            reader = VideoReader(vpath)
            count = 0
            for idx, frame in reader.iter_frames(max_frames=args.max_frames * args.step if args.max_frames else None):
                if idx % args.step != 0:
                    continue
                enhanced_frame, report = manager.process_frame(frame, mode=args.mode)
                out_file = os.path.join(out_folder, f"frame_{idx:06d}_enhanced.{args.format}")
                cv2.imwrite(out_file, enhanced_frame)
                count += 1
                if count % 10 == 0:
                    print(f"  Saved {count} enhanced frames...")
            reader.release()
            print(f"  Finished: {count} enhanced frames saved to {out_folder}")
    else:
        extractor = FrameExtractor(output_root=args.output_dir)
        if args.video:
            print(f"Extracting raw frames from {args.video}...")
            extracted = extractor.extract_from_video(
                args.video,
                step=args.step,
                max_frames=args.max_frames,
                image_format=args.format,
            )
            print(f"Successfully extracted {len(extracted)} frames to {args.output_dir}/")
        else:
            print(f"Extracting raw frames from all videos in {args.video_dir}...")
            results = extractor.extract_all_in_folder(
                video_folder=args.video_dir,
                step=args.step,
                max_frames_per_video=args.max_frames,
                image_format=args.format,
            )
            total = sum(len(f) for f in results.values())
            print(f"Extracted a total of {total} frames across {len(results)} videos.")


if __name__ == "__main__":
    main()
