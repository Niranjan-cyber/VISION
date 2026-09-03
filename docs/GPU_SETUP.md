# VISION — GPU (CUDA) Acceleration Setup

**Optional.** The system works fully on CPU with zero setup (`pip install -r requirements.txt`
is CPU-only and portable to machines with no NVIDIA GPU — see [README.md](../README.md)).
This document is only for enabling CUDA acceleration on a machine with a
supported NVIDIA GPU, as was done for the demo laptop (RTX 4060 Laptop GPU).

## What actually gets GPU-accelerated, and why the rest doesn't

| Component | Device | Why |
| :--- | :--- | :--- |
| YOLO11n (detection) | **CUDA** (auto) | `ultralytics` runs on torch; a CUDA-enabled torch build moves inference to the GPU. |
| InsightFace W600K-R50 (face recognition) | **CUDA** (auto) | Raw `onnxruntime.InferenceSession`; `CUDAExecutionProvider` moves inference to the GPU. |
| YuNet (face detection) | **CPU always** | Diagnosed directly: this project's `opencv-python` wheel has `cv2.cuda.getCudaEnabledDeviceCount() == 0`, and its DNN "new graph engine" explicitly does not support backend/target selection — requesting `DNN_BACKEND_CUDA`/`DNN_TARGET_CUDA` emits *"Back-ends/Targets are not supported by the new graph engine for now"* and silently no-ops. Building a custom CUDA-enabled OpenCV from source was judged not worth the fragility for an SIH MVP — see the note in [src/core/device.py](../src/core/device.py). |
| ByteTrack, EventEngine | **CPU always** | Lightweight bookkeeping (Kalman filter + Hungarian matching, deterministic rules) — no GPU implementation in the libraries used, and not worth accelerating. |

## Exact install steps (what was actually run)

These versions were chosen empirically for **this** machine (driver 572.83,
max CUDA 12.8 per `nvidia-smi`) — see the "why these versions" notes below
before blindly copying version numbers onto different hardware.

```bash
# 1. CUDA-enabled torch/torchvision (self-contained wheels, no system CUDA Toolkit needed)
pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126" "torchvision==0.28.0+cu126"

# 2. onnxruntime-gpu — MUST replace the CPU-only onnxruntime package (same Python import name, conflicting files)
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu==1.22.0"

# 3. CUDA 12 runtime libraries onnxruntime-gpu's CUDAExecutionProvider needs at load time
#    (pip-only, no system CUDA Toolkit install required)
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 \
            nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nvjitlink-cu12
```

### Why these exact versions (diagnosed, not guessed)

- **torch `cu126`, not `cu130`/`cu132`**: `nvidia-smi` reports the installed driver (572.83)
  supports up to CUDA **12.8**. Newer CUDA 13.x wheels risk exceeding what this driver
  supports; 12.6 is comfortably within range.
- **`onnxruntime-gpu==1.22.0`, not the latest `1.29.0`**: tried `1.29.0` first — it failed
  to load its CUDA provider DLL, reporting *"Require cuDNN 9.\* and CUDA 13.\*"* (confirmed
  by the exact missing DLL name, `cublasLt64_13.dll`). `1.22.0` reported *"Require cuDNN 9.\*
  and CUDA 12.\*"* instead (`cublasLt64_12.dll`) — matching what the driver actually supports.
  If you need a newer onnxruntime-gpu later, check what CUDA major version its error message
  requests (attempt CUDA-provider session creation and read the message) before assuming it matches.
- **The 6 `nvidia-*-cu12` packages**: onnxruntime's `onnxruntime_providers_cuda.dll` transitively
  depends on cuBLAS, cuDNN, cuFFT, cuRAND, cuSPARSE, cuSOLVER, and nvJitLink — discovered one
  missing-DLL error at a time rather than assumed. Installing all of them up front avoids the
  same discovery loop.

### The DLL-search-path gotcha

Installing the `nvidia-*-cu12` pip packages alone is **not enough** — their DLLs land in
`site-packages/nvidia/*/bin`, which isn't on Windows' default DLL search path.
`os.add_dll_directory()` (the "modern", documented fix) was tried first and **did not work**
for onnxruntime's provider-bridge loader specifically (confirmed: a direct `ctypes.WinDLL` load
of the same DLL succeeded with directories registered that way, while onnxruntime's own load of
it still failed). Prepending those directories to the `PATH` environment variable **does** work
and is what `src/core/device.py`'s `_register_pip_cuda_dll_dirs()` actually does — this is
handled automatically the first time GPU providers are requested; no manual step needed once
the packages above are installed.

## Verifying it worked

```bash
python -m src.main --device cuda --debug-face-matching
```
Look for the startup banner:
```
VISION Hardware Configuration
-------------------------------
Requested device       : cuda
GPU                    : NVIDIA GeForce RTX 4060 Laptop GPU
YOLO (detection)       : CUDA
Face Recognition       : CUDAExecutionProvider
Face Detection (YuNet) : CPU (OpenCV build has no CUDA DNN support)
Tracking (ByteTrack)   : CPU
Event Engine           : CPU
```
If `Face Recognition` shows `CPUExecutionProvider` despite requesting `cuda`, a `[WARNING]`
line explains exactly why (missing provider, or onnxruntime granted a different provider than
requested) — the code never silently claims GPU it isn't actually using.

## `--device` flag / `VISION_DEVICE` env var

| Value | Behavior |
| :--- | :--- |
| `auto` (default) | CUDA if `torch.cuda.is_available()` / `onnxruntime` reports `CUDAExecutionProvider` — else CPU. Never errors due to missing GPU. |
| `cuda` | Forces the attempt; falls back to CPU per-component with a `[WARNING]` if unavailable. |
| `cpu` | Forces CPU regardless of hardware — useful for a clean A/B comparison. |

CLI: `--device auto|cuda|cpu`. Backend: `VISION_DEVICE` env var, same values, applies to every camera.
