"""GPU selection and evidence shared by the controller and stage processes.

Importing this module does not import torch: the control environment must not
decide capabilities on behalf of the isolated model environments.
"""
from __future__ import annotations

import importlib.metadata
import importlib
import math
import os
import platform
import resource
import threading
import time
import subprocess


CUDA_DEFAULT_GPU = "0"
PROBE_TIMEOUT = 90


def cuda_environment(gpu):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu if gpu is not None else CUDA_DEFAULT_GPU)}


def resolve_device(requested="auto", *, gpu=None):
    if requested not in {"auto", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested in {"auto", "cuda"}:
        if gpu is None:
            import torch
            cuda_available = torch.cuda.is_available()
        else:
            import sys
            try:
                result = subprocess.check_output(
                    [sys.executable, "-c", "import torch; print(int(torch.cuda.is_available()))"],
                    env=cuda_environment(gpu), text=True, timeout=PROBE_TIMEOUT)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("CUDA device probe timed out") from error
            cuda_available = result.strip().splitlines()[-1:] == ["1"]
        if cuda_available:
            return "cuda"
    if requested in {"auto", "mps"}:
        import torch
        if torch.backends.mps.is_available():
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
                raise RuntimeError("MPS validation requires PYTORCH_ENABLE_MPS_FALLBACK=0 before importing torch")
            return "mps"
    raise RuntimeError(f"Requested GPU {requested!r} unavailable; CPU fallback is disabled")


def memory_budget(value, device):
    budget = float(value) if value is not None else (16.0 if device == "mps" else 80.0)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("memory budget must be positive and finite")
    return budget


def probe(requested="auto", environment=None):
    import torch
    device = resolve_device(requested)
    imports = {
        "control": ["numpy", "scipy", "soundfile", "yaml", "huggingface_hub"],
        "sheetsage2": ["transformers", "torchaudio", "safetensors", "pretty_midi", "mir_eval"],
        "yue2": ["yue2", "transformers", "accelerate", "tiktoken", "soundfile"],
        "separator": ["audio_separator.separator", "onnxruntime", "torchaudio"],
        "seed-vc": ["transformers", "torchaudio", "librosa", "yaml", "munch", "inference"],
    }
    if environment is not None:
        for name in imports[environment]:
            importlib.import_module(name)
    x = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    actual = (x @ x.T).cpu()
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torch.testing.assert_close(actual, expected @ expected.T)
    return {**runtime_identity(), "device": device, "matmul": "passed"}


def runtime_identity():
    """Describe installed packages without importing torch or initializing a device."""
    packages = {item.metadata["Name"].lower().replace("_", "-"): item.version
                for item in importlib.metadata.distributions() if item.metadata["Name"]}
    return {"packages": packages, "python": platform.python_version(),
            "machine": platform.machine(), "platform": platform.platform()}


def assert_model_device(model, device):
    tensors = list(model.parameters()) + list(model.buffers())
    wrong = sorted({str(t.device) for t in tensors if t.device.type != device})
    if wrong:
        raise RuntimeError(f"Model tensors on {wrong}, expected {device}")


class MemoryMonitor:
    """Sample allocator usage; retain evidence even when inference raises.

    RSS is process-wide and includes CPU tensors. It is not added to MPS
    allocations, which share the same physical memory on Apple Silicon.
    """
    def __init__(self, device, budget):
        self.device, self.budget = device, budget
        self.started = time.monotonic()
        self.effective_budget = budget
        self.stop = threading.Event()
        self.samples = {"mps_allocated_peak_bytes": 0, "mps_driver_peak_bytes": 0}

    def sample(self):
        import torch
        if self.device == "mps":
            for key, value in (("mps_allocated_peak_bytes", torch.mps.current_allocated_memory()),
                               ("mps_driver_peak_bytes", torch.mps.driver_allocated_memory())):
                self.samples[key] = max(self.samples[key], value)

    def watch(self):
        while not self.stop.wait(0.2):
            self.sample()

    def __enter__(self):
        import torch
        if self.device == "mps":
            recommended = torch.mps.recommended_max_memory()
            self.effective_budget = min(self.budget, recommended / 2**30)
            torch.mps.set_per_process_memory_fraction(self.effective_budget * 2**30 / recommended)
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sample()

    def report(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {**self.samples, "rss_peak_bytes": rss if platform.system() == "Darwin" else rss * 1024,
                "memory_budget_gib": self.budget, "device": self.device,
                "effective_mps_budget_gib": self.effective_budget if self.device == "mps" else None,
                "wall_seconds": time.monotonic() - self.started, "sample_interval_s": 0.2}


class ForwardAudit:
    """Observe actual learned-layer calls in a dedicated inference process."""
    def __init__(self, device):
        self.device = device
        self.layers = {}

    def check(self, module, inputs):
        import torch
        parameters = list(module.parameters(recurse=False))
        if not parameters:
            return
        for tensor in parameters:
            if tensor.device.type != self.device:
                raise RuntimeError(f"{type(module).__name__} forward on {tensor.device}, expected {self.device}")
        for tensor in inputs:
            if isinstance(tensor, torch.Tensor) and tensor.device.type != self.device:
                raise RuntimeError(f"{type(module).__name__} input on {tensor.device}, expected {self.device}")
        key = f"{type(module).__name__}:{parameters[0].dtype}:{self.device}"
        self.layers[key] = self.layers.get(key, 0) + 1

    def __enter__(self):
        from torch.nn.modules.module import register_module_forward_pre_hook
        self.handle = register_module_forward_pre_hook(self.check)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
