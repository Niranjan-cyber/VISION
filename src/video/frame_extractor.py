"""
Frame extractor module for sampling and saving video frames to disk.
"""

import os
import cv2
from typing import List, Optional
from tqdm import tqdm
from .reader import VideoReader


class FrameExtractor:
    """
    Utility for extracting and exporting frames from video files into structured frame directories.
    """

    def __init__(self, output_root: str = "frames"):
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

    def extract_from_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        step: int = 1,
        max_frames: Optional[int] = None,
        image_format: str = "png",
        show_progress: bool = True,
    ) -> List[str]:
        """
        Extract frames from a specific video file.

        Args:
            video_path: Path to the input video file.
            output_dir: Target subdirectory. Defaults to output_root/{video_name_without_ext}.
            step: Extract every `step`-th frame (1 = every frame).
            max_frames: Maximum total frames to extract.
            image_format: 'png' (lossless) or 'jpg'.
            show_progress: Display tqdm progress bar.

        Returns:
            List of saved image filepaths.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if output_dir is None:
            output_dir = os.path.join(self.output_root, video_name)
        os.makedirs(output_dir, exist_ok=True)

        reader = VideoReader(video_path)
        saved_paths = []
        total_to_process = reader.total_frames
        if max_frames is not None:
            total_to_process = min(total_to_process, max_frames * step)

        pbar = tqdm(total=total_to_process, desc=f"Extracting {video_name}", disable=not show_progress)
        
        extracted_count = 0
        for frame_idx, frame in reader.iter_frames():
            if frame_idx % step == 0:
                out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.{image_format}")
                cv2.imwrite(out_path, frame)
                saved_paths.append(out_path)
                extracted_count += 1
                if max_frames is not None and extracted_count >= max_frames:
                    pbar.update(1)
                    break
            pbar.update(1)

        pbar.close()
        reader.release()
        return saved_paths

    def extract_all_in_folder(
        self,
        video_folder: str = "videos",
        step: int = 1,
        max_frames_per_video: Optional[int] = None,
        image_format: str = "png",
    ) -> dict:
        """
        Extract frames for all MP4 / AVI videos found in a directory.
        """
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f"Folder not found: {video_folder}")

        video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".flv"}
        results = {}
        for fname in sorted(os.listdir(video_folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in video_extensions:
                vpath = os.path.join(video_folder, fname)
                extracted = self.extract_from_video(
                    vpath,
                    step=step,
                    max_frames=max_frames_per_video,
                    image_format=image_format,
                )
                results[fname] = extracted
        return results
