"""
Unit tests for VideoReader, VideoWriter, and FrameExtractor.
"""

import os
import cv2
import numpy as np
import pytest
import tempfile

from src.video.reader import VideoReader
from src.video.writer import VideoWriter
from src.video.frame_extractor import FrameExtractor


@pytest.fixture
def sample_video_path(tmp_path):
    video_path = str(tmp_path / "test_sample.mp4")
    w, h = 320, 240
    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for i in range(15):
        frame = np.full((h, w, 3), fill_value=(i * 15, 120, 200), dtype=np.uint8)
        cv2.circle(frame, (160, 120), 30, (255, 255, 255), -1)
        out.write(frame)

    out.release()
    return video_path


def test_video_reader(sample_video_path):
    reader = VideoReader(sample_video_path)
    assert reader.width == 320
    assert reader.height == 240
    assert reader.fps > 0
    assert reader.total_frames == 15

    frames = list(reader)
    assert len(frames) == 15
    assert frames[0].shape == (240, 320, 3)
    reader.release()


def test_video_writer(tmp_path):
    out_path = str(tmp_path / "output.mp4")
    writer = VideoWriter(out_path, fps=25.0, frame_size=(320, 240))

    for _ in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write_frame(frame)

    writer.release()
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_comparison_frame_generation():
    orig = np.zeros((240, 320, 3), dtype=np.uint8)
    enh = np.ones((240, 320, 3), dtype=np.uint8) * 255

    comp = VideoWriter.create_comparison_frame(orig, enh, metrics_text="Test Metric")
    assert comp.shape[0] == 240
    assert comp.shape[1] % 16 == 0  # H.264 macroblock divisible
    assert comp.shape[2] == 3


def test_frame_extractor(sample_video_path, tmp_path):
    extractor = FrameExtractor(output_root=str(tmp_path / "frames"))
    extracted = extractor.extract_from_video(sample_video_path, step=2)
    assert len(extracted) == 8
    for fpath in extracted:
        assert os.path.exists(fpath)
