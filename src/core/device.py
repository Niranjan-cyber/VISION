"""
Device / execution-provider selection for VISION's GPU-accelerable
components (YOLO11n via torch, InsightFace W600K-R50 via onnxruntime).

Single source of truth so `--device auto|cuda|cpu` behaves identically for
both, and so the startup hardware banner reports exactly what each
component actually resolved to rather than what was merely requested.

YuNet (OpenCV FaceDetectorYN) and ByteTrack/EventEngine are deliberately
NOT wired here. Diagnosed directly: this project's opencv-python build has
zero CUDA devices compiled in (`cv2.cuda.getCudaEnabledDeviceCount() == 0`)
and its DNN "new graph engine" explicitly does not support backend/target
selection at all — requesting `DNN_BACKEND_CUDA`/`DNN_TARGET_CUDA` emits
"Back-ends/Targets are not supported by the new graph engine for now" and
silently no-ops. YuNet stays CPU-only by evidence, not by omission.
ByteTrack/EventEngine are lightweight CPU bookkeeping with no GPU
implementation in the libraries this project uses.
"""
import glob
import os
import sys
from typing import List, Optional

VALID_DEVICE_PREFS = {"auto", "cuda", "cpu"}

_torch_cuda_available: Optional[bool] = None
_gpu_name: Optional[str] = None
_ort_cuda_available: Optional[bool] = None
_dll_dirs_registered = False


def _register_pip_cuda_dll_dirs() -> None:
    """
    onnxruntime-gpu's CUDAExecutionProvider is a separate DLL
    (onnxruntime_providers_cuda.dll) that depends on cuBLAS/cuDNN/cuFFT/
    cuRAND/cuSPARSE/cuSOLVER/nvrtc/the CUDA runtime at load time. When
    those come from pip packages (nvidia-cublas-cu12, nvidia-cudnn-cu12,
    etc.) rather than a system-wide CUDA Toolkit install, their DLLs sit
    in site-packages/nvidia/*/bin — not on the default Windows DLL search
    path — so onnxruntime fails to find them even though they're
    installed.

    Empirically verified two candidate fixes on this project's stack:
    os.add_dll_directory() (the "modern", documented approach) does NOT
    work here — onnxruntime's provider-bridge loader still fails to find
    the DLLs even with directories registered that way, confirmed by a
    direct ctypes.WinDLL load of the same DLL succeeding while
    onnxruntime's load of it still failed. Prepending the directories to
    the PATH environment variable *does* work and is what onnxruntime's
    own loader actually respects, so that's what this function does.
    No-op if the packages aren't installed (nothing matches the glob).
    """
    global _dll_dirs_registered
    if _dll_dirs_registered:
        return
    _dll_dirs_registered = True

    try:
        import site
        search_roots = set(site.getsitepackages())
        try:
            search_roots.add(site.getusersitepackages())
        except Exception:
            pass
    except Exception:
        search_roots = {os.path.join(sys.prefix, "Lib", "site-packages")}

    found_dirs = []
    for root in search_roots:
        found_dirs.extend(glob.glob(os.path.join(root, "nvidia", "*", "bin")))

    if found_dirs:
        os.environ["PATH"] = os.pathsep.join(found_dirs) + os.pathsep + os.environ.get("PATH", "")


def torch_cuda_available() -> bool:
    """Whether torch actually sees a usable CUDA device (cached after first check)."""
    global _torch_cuda_available, _gpu_name
    if _torch_cuda_available is None:
        try:
            import torch
            _torch_cuda_available = bool(torch.cuda.is_available())
            if _torch_cuda_available:
                _gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            _torch_cuda_available = False
    return _torch_cuda_available


def gpu_name() -> Optional[str]:
    """The detected GPU name, if torch can see one — else None."""
    torch_cuda_available()
    return _gpu_name


def ort_cuda_available() -> bool:
    """Whether the installed onnxruntime build exposes CUDAExecutionProvider."""
    global _ort_cuda_available
    if _ort_cuda_available is None:
        try:
            import onnxruntime as ort
            _ort_cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            _ort_cuda_available = False
    return _ort_cuda_available


def resolve_yolo_device(device_pref: str) -> str:
    """
    device_pref: 'auto' | 'cuda' | 'cpu'.
    Returns the torch device string YOLODetector should actually use.
    'cuda' requested but unavailable -> falls back to CPU with a warning
    (never silently claims GPU it doesn't have).
    """
    if device_pref not in VALID_DEVICE_PREFS:
        device_pref = "auto"
    if device_pref == "cpu":
        return "cpu"
    if device_pref == "cuda" and not torch_cuda_available():
        print(
            "[WARNING] --device cuda requested but CUDA is not available to torch "
            "in this environment; YOLO will run on CPU.",
            file=sys.stderr,
        )
        return "cpu"
    if device_pref == "auto" and not torch_cuda_available():
        return "cpu"
    return "cuda:0"


def resolve_ort_providers(device_pref: str) -> List[str]:
    """
    device_pref: 'auto' | 'cuda' | 'cpu'.
    Returns the onnxruntime provider priority list InferenceSession should
    request. The caller must still check session.get_providers() afterward
    to see which one onnxruntime actually picked — this function only
    decides what to *ask* for.
    """
    if device_pref not in VALID_DEVICE_PREFS:
        device_pref = "auto"
    if device_pref == "cpu":
        return ["CPUExecutionProvider"]
    if device_pref == "cuda" and not ort_cuda_available():
        print(
            "[WARNING] --device cuda requested but this onnxruntime build has no "
            "CUDAExecutionProvider (CPU-only onnxruntime installed?); face "
            "recognition will run on CPU.",
            file=sys.stderr,
        )
        return ["CPUExecutionProvider"]
    if device_pref == "auto" and not ort_cuda_available():
        return ["CPUExecutionProvider"]
    _register_pip_cuda_dll_dirs()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


_banner_printed = False


def print_hardware_banner(device_pref: str, yolo_device: str, face_provider: str, force: bool = False) -> None:
    """Prints the actual resolved hardware configuration — never claims GPU
    acceleration a component didn't confirm it's using. Multiple cameras
    share identical hardware, so this prints once per process by default
    (pass force=True to print again regardless)."""
    global _banner_printed
    if _banner_printed and not force:
        return
    _banner_printed = True

    gpu = gpu_name() or "none detected"
    rows = [
        ("Requested device", device_pref),
        ("GPU", gpu),
        ("YOLO (detection)", "CUDA" if yolo_device.startswith("cuda") else "CPU"),
        ("Face Recognition", face_provider),
        ("Face Detection (YuNet)", "CPU (OpenCV build has no CUDA DNN support)"),
        ("Tracking (ByteTrack)", "CPU"),
        ("Event Engine", "CPU"),
    ]
    label_width = max(len(label) for label, _ in rows)
    lines = ["VISION Hardware Configuration", "-" * 31]
    lines += [f"{label:<{label_width}} : {value}" for label, value in rows]
    print("\n".join(lines), file=sys.stderr)
