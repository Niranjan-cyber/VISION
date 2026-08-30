"""
Main Command-Line Interface for SIH 2026 CCTV Intelligent Video Enhancement Pipeline.
Processes CCTV footage, applies dynamic quality routing, and produces enhanced videos & comparison analytics.
"""

import os
import sys
import argparse
import time

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.enhancement.manager import EnhancementManager


def parse_args():
    parser = argparse.ArgumentParser(description="SIH 2026: Intelligent CCTV Video Enhancement Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="videos/VIRAT_S_010205_04_000545_000576.mp4",
        help="Input video file path or directory or RTSP stream URL",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="enhanced_videos/pipeline",
        help="Directory to save enhanced videos",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "zero_dce", "realesrgan", "basicvsr", "rvrt", "all"],
        help="Enhancement mode (auto = Quality Analyzer dynamic routing)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="Generate side-by-side comparison video with live HUD overlay",
    )
    parser.add_argument(
        "--no_compare",
        dest="compare",
        action="store_false",
        help="Do not generate comparison video",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to process (useful for rapid testing)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame step interval (default: 1, every frame)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Also save all enhanced frames as individual images in frames/enhanced/<video_name>/",
    )
    parser.add_argument(
        "--save_frames_dir",
        type=str,
        default=None,
        help="Custom directory to save enhanced frame images",
    )
    return parser.parse_args()


def process_single_video(manager: EnhancementManager, video_path: str, args):
    video_name = os.path.basename(video_path)
    base_name, ext = os.path.splitext(video_name)

    mode_subfolder = args.mode if args.mode != "auto" else "pipeline"
    out_dir = os.path.join("enhanced_videos", mode_subfolder)
    os.makedirs(out_dir, exist_ok=True)

    out_video_path = os.path.join(out_dir, f"{base_name}_enhanced{ext}")
    comp_video_path = os.path.join(out_dir, f"{base_name}_comparison{ext}") if args.compare else None

    # Setup frame save directory if requested
    frames_dir = None
    if args.save_frames or args.save_frames_dir:
        frames_dir = args.save_frames_dir or os.path.join("frames", "enhanced", base_name)
        os.makedirs(frames_dir, exist_ok=True)

    print(f"\n[START] Processing: {video_name}")
    print(f"  Input:       {video_path}")
    print(f"  Output:      {out_video_path}")
    if comp_video_path:
        print(f"  Comparison:  {comp_video_path}")
    if frames_dir:
        print(f"  Frames Dir:  {frames_dir}")
    print(f"  Mode:        {args.mode.upper()}")
    if args.max_frames:
        print(f"  Max Frames:  {args.max_frames}")

    summary = manager.process_video(
        input_path=video_path,
        output_path=out_video_path,
        mode=args.mode,
        create_comparison=args.compare,
        comparison_output_path=comp_video_path,
        max_frames=args.max_frames,
        step=args.step,
        show_progress=True,
        save_frames_dir=frames_dir,
    )

    print(f"[COMPLETED] {video_name}:")
    print(f"  Frames:      {summary['frames_processed']}")
    print(f"  Time:        {summary['elapsed_seconds']} s ({summary['processing_fps']} FPS)")
    print(f"  Diagnostics: {summary['quality_distribution']}")
    return summary


def main():
    args = parse_args()

    print("=" * 75)
    print("      SIH 2026: INTELLIGENT CCTV VIDEO ENHANCEMENT SYSTEM")
    print("=" * 75)

    manager = EnhancementManager(device=args.device)

    # Check if input is directory or file
    if os.path.isdir(args.input):
        video_exts = {".mp4", ".avi", ".mkv", ".mov"}
        video_files = [
            os.path.join(args.input, f)
            for f in sorted(os.listdir(args.input))
            if os.path.splitext(f)[1].lower() in video_exts
        ]
        if not video_files:
            print(f"No video files found in directory: {args.input}")
            return

        print(f"Found {len(video_files)} videos in {args.input}. Starting batch processing...")
        summaries = []
        for vf in video_files:
            summary = process_single_video(manager, vf, args)
            summaries.append(summary)

        print("\n" + "=" * 75)
        print("BATCH PROCESSING SUMMARY:")
        for s in summaries:
            print(f"  - {os.path.basename(s['input_video'])}: {s['frames_processed']} frames in {s['elapsed_seconds']}s ({s['processing_fps']} fps)")
        print("=" * 75)

    elif os.path.isfile(args.input):
        process_single_video(manager, args.input, args)
    else:
        # RTSP stream or camera index
        print(f"Connecting to stream: {args.input}")
        out_video_path = os.path.join(args.output_dir, "stream_enhanced.mp4")
        manager.process_video(
            input_path=args.input,
            output_path=out_video_path,
            mode=args.mode,
            create_comparison=args.compare,
            max_frames=args.max_frames,
            step=args.step,
        )


if __name__ == "__main__":
    main()
