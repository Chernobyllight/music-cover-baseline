#!/usr/bin/env python3
"""Run the pinned short score through AR, NAR and VAE, retaining each result.

This diagnostic uses a short generation budget; production cover budgets are
unchanged. Run with envs/yue2/bin/python and MPS CPU fallback disabled.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from music_cover.runtime import ForwardAudit, MemoryMonitor, probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from yue2 import YuE2Pipeline
    import numpy as np
    import soundfile as sf
    report = {"environment": probe("mps"), "stages": {}}
    monitor = MemoryMonitor("mps", 16)
    audit = ForwardAudit("mps")
    try:
        with monitor, audit, YuE2Pipeline.from_pretrained(
            str(ROOT / "models/YuE2-3B"), vae=str(ROOT / "models/YuE2-Vae"),
            device="mps", backend="torch-eager", quantization="none",
            memory_budget_gib=16, vae_core_frames=512, local_files_only=True,
            offload_ar=True,
        ) as pipeline:
            request = json.loads((ROOT / "third_party/YuE/examples/song.json").read_text())
            request["abc"] = (ROOT / "third_party/YuE/examples/melody.abc").read_text()
            request["cot"] = "melody"
            plan = pipeline.plan(**request)
            plan.save(args.output / "plan")
            start = time.monotonic()
            semantic = pipeline.generate_semantic(plan, sampling={"max_tokens": 2048})
            np.save(args.output / "semantic.npy", np.asarray(semantic.tokens, dtype=np.int32))
            report["stages"]["semantic"] = {"seconds": time.monotonic() - start, "truncated": semantic.truncated}
            if semantic.truncated:
                raise RuntimeError("Short-score semantic generation truncated")
            start = time.monotonic()
            latents = pipeline.synthesize(semantic)
            np.save(args.output / "latents.npy", latents)
            if not np.isfinite(latents).all():
                raise RuntimeError("Non-finite NAR latents")
            report["stages"]["synthesize"] = {"seconds": time.monotonic() - start}
            start = time.monotonic()
            audio = pipeline.decode(latents)
            sf.write(args.output / "audio.wav", audio, 48000, subtype="PCM_24")
            report["stages"]["decode"] = {"seconds": time.monotonic() - start}
            report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report.update(memory=monitor.report(), forward_layers=audit.layers)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
