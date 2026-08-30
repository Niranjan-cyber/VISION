# SIH 2026: Intelligent CCTV Video Enhancement System

An end-to-end, modular deep learning pipeline for real-time and offline CCTV surveillance footage enhancement. The system performs multi-dimensional video quality diagnostics (detecting low-light, blur, low-resolution, and poor dynamic range) and dynamically routes degraded frames through specialized neural network models (**Zero-DCE++**, **RVRT**, **Real-ESRGAN**, **BasicVSR++**) to generate crystal-clear output videos with side-by-side analytical comparisons.

---

## 🏛️ System Architecture Flowchart

```
                 EXISTING CCTV / RTSP STREAM
                              │
                              ▼
                     ┌──────────────────┐
                     │ OpenCV + FFmpeg  │
                     │   Video Reader   │
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │ Quality Analyzer │
                     │ (Dark, Blur, SR) │
                     └────────┬─────────┘
                              │
                     ┌────────┴────────┐
                     │                 │
                 Good Quality      Poor Quality
                     │                 │
                     │        ┌────────┴─────────────────┐
                     │        │                 │        │
                     │    Low-light           Blur    Low-res
                     │        │                 │        │
                     │    Zero-DCE++          RVRT    Real-ESRGAN / BasicVSR++
                     │        │                 │        │
                     │        └────────┬────────┘────────┘
                     │                 │
                     └────────┬────────┘
                              ▼
                       Enhanced Video
                + Side-by-Side Comparison HUD
```

---

## 📁 Project Directory Structure

```
SIH_2026/
│
├── videos/                                # Raw CCTV input footage (e.g. VIRAT dataset)
│   ├── VIRAT_S_010205_04_000545_000576.mp4
│   └── ...
│
├── frames/                                # Extracted video frames per clip
│   ├── VIRAT_S_010205.../
│   └── ...
│
├── enhanced_videos/                       # Enhanced video outputs
│   ├── zero_dce/                          # Standalone low-light enhanced videos
│   ├── realesrgan/                        # Standalone super-resolved videos
│   ├── basicvsr/                          # Video recurrent super-resolved videos
│   ├── rvrt/                              # Standalone deblurred videos
│   └── pipeline/                          # Full intelligent adaptive pipeline outputs
│
├── models/                                # Pretrained model weights
│   ├── zero_dce/weights/
│   ├── realesrgan/weights/
│   ├── basicvsr/weights/
│   └── rvrt/weights/
│
├── src/                                   # Core Python source code
│   │
│   ├── video/                             # Video I/O layer
│   │   ├── __init__.py
│   │   ├── reader.py                      # Multi-threaded VideoCapture & RTSP streaming
│   │   ├── frame_extractor.py             # Batch frame extraction
│   │   └── writer.py                      # Multi-codec video writer & side-by-side HUD creator
│   │
│   ├── quality/                           # Quality diagnostic layer
│   │   ├── __init__.py
│   │   ├── analyzer.py                    # Master QualityAnalyzer & QualityReport
│   │   ├── brightness.py                  # Luminance & dark-pixel scoring
│   │   ├── blur.py                        # Laplacian variance, Tenengrad & FFT blur
│   │   ├── contrast.py                    # RMS contrast & dynamic range
│   │   └── resolution.py                  # Edge density & spatial resolution
│   │
│   └── enhancement/                       # Deep learning restoration models
│       ├── __init__.py
│       ├── base.py                        # BaseEnhancer interface
│       ├── manager.py                     # Adaptive routing EnhancementManager
│       │
│       ├── low_light/
│       │   └── zero_dce.py                # Zero-DCE & Zero-DCE++ curve estimation
│       │
│       ├── low_resolution/
│       │   ├── realesrgan.py              # Real-ESRGAN compact tile-based SR
│       │   └── basicvsr.py                # BasicVSR++ recurrent video SR
│       │
│       └── blur/
│           └── rvrt.py                    # RVRT & deep motion deblurring
│
├── tests/                                 # Pytest unit tests
│   ├── test_video/
│   ├── test_quality/
│   └── test_enhancement/
│
├── notebooks/                             # Interactive Jupyter Notebooks
│   ├── video_analysis.ipynb               # Video source exploration & metadata
│   ├── quality_analysis.ipynb             # Frame degradation diagnostics & plots
│   └── enhancement_comparison.ipynb       # Before-and-after visual comparisons
│
├── scripts/                               # CLI execution scripts
│   ├── extract_frames.py                  # CLI frame extraction
│   ├── download_weights.py                # Pretrained weights downloader
│   └── run_pipeline.py                    # Full CCTV enhancement pipeline CLI
│
├── requirements.txt                       # Project dependencies
├── README.md                              # Documentation
└── .gitignore                             # Git ignore rules
```

---

## ⚡ Pretrained Models & Architectures

1. **Low-Light Enhancement**: [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) / [Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension)
   - Deep Curve Estimation network.
   - Formulates light enhancement as high-order curve parameter estimation $\mathcal{A}$, performing zero-reference non-linear illumination brightening without overexposing highlights.
2. **Super-Resolution**: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) & [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus)
   - Real-ESRGAN: Compact SRVGG / RRDB network with memory-safe overlapping tile inference for single-frame license plate, face, and texture restoration.
   - BasicVSR++: Recurrent video super-resolution exploiting temporal redundancy across neighboring frames.
3. **Deblurring & Motion Restoration**: [RVRT](https://github.com/JingyunLiang/RVRT)
   - Recurrent Video Restoration Transformer for motion blur and defocus deblurring in high-speed surveillance scenes.

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install required packages:
```bash
pip install -r requirements.txt
```

### 2. Download Pretrained Model Weights

Download or initialize model checkpoints for Zero-DCE, Real-ESRGAN, BasicVSR, and RVRT:
```bash
python scripts/download_weights.py
```

### 3. Extract Frames from CCTV Footage

Extract frames from a specific CCTV video or entire `videos/` folder:
```bash
# Extract from a single video
python scripts/extract_frames.py --video videos/VIRAT_S_010205_04_000545_000576.mp4 --step 2

# Extract from all videos in videos/ folder
python scripts/extract_frames.py --video_dir videos/ --step 5 --max_frames 50
```

### 4. Run the Intelligent Enhancement Pipeline via CLI

Run adaptive quality-driven video enhancement:
```bash
# Run intelligent auto-routing pipeline on sample video
python scripts/run_pipeline.py --input videos/VIRAT_S_010205_04_000545_000576.mp4 --mode auto --compare

# Run on all videos in videos/ directory
python scripts/run_pipeline.py --input videos/ --mode auto --max_frames 60

# Run specific enhancement model (e.g. Zero-DCE low-light only)
python scripts/run_pipeline.py --input videos/VIRAT_S_010205_04_000545_000576.mp4 --mode zero_dce
```

The enhanced video and side-by-side comparison video will be saved under `enhanced_videos/pipeline/`.

---

## 🧪 Running Unit Tests

Run the complete automated test suite:
```bash
pytest -v
```

---

## 📊 Jupyter Notebooks

Launch Jupyter to explore interactive analysis and benchmark notebooks:
```bash
jupyter notebook
```
- [`notebooks/video_analysis.ipynb`](file:///c:/Users/shubh/Documents/SIH_2026/notebooks/video_analysis.ipynb): Inspect CCTV frame properties.
- [`notebooks/quality_analysis.ipynb`](file:///c:/Users/shubh/Documents/SIH_2026/notebooks/quality_analysis.ipynb): Analyze degradation metric trajectories.
- [`notebooks/enhancement_comparison.ipynb`](file:///c:/Users/shubh/Documents/SIH_2026/notebooks/enhancement_comparison.ipynb): Side-by-side before/after model comparisons.
