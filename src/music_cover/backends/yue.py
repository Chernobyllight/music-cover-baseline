from __future__ import annotations
import hashlib
import time
import json
import math
from pathlib import Path
from ..common.io import identity as checkpoint_identity, write_json
from ..common.paths import ROOT

def cover_token_budget(score_seconds, prefix_tokens, context=24576, tokens_per_second=25):
    """Leave room for a complete score and ending, without shortening for context."""
    if not math.isfinite(score_seconds) or score_seconds <= 0:
        raise ValueError("Score duration must be positive and finite")
    margin = max(30.0, score_seconds * 0.15)
    maximum = max(9000, math.ceil((score_seconds + margin) * tokens_per_second / 1000) * 1000)
    if prefix_tokens + maximum > context:
        raise ValueError(
            f"Score needs {maximum} semantic tokens plus {prefix_tokens} prompt tokens, "
            f"exceeding YuE2 context {context}; refusing to silently truncate the song")
    return {"score_duration_s": score_seconds, "ending_margin_s": margin,
            "semantic_tokens_per_second": tokens_per_second, "semantic_max_tokens": maximum,
            "audio_ceiling_s": maximum / tokens_per_second, "prefix_tokens": prefix_tokens,
            "context_tokens": context}

def cover(args):
    cover_start = time.monotonic()
    import numpy as np
    from yue2 import YuE2Pipeline, SemanticResult, SongResult
    from yue2.storage import identity
    from music_cover.pitch import abc_tools
    request = json.loads(Path(args.request).read_text())
    request["abc"] = Path(args.score).read_text()
    request["cot"] = "melody"
    score = abc_tools().parse(request["abc"])
    score_seconds = float(score.voices["Vocal"].time * 60 / score.bpm)
    with YuE2Pipeline.from_pretrained(
        str(ROOT / "models/YuE2-3B"), vae=str(ROOT / "models/YuE2-Vae"),
        device=args.device, local_files_only=True, memory_budget_gib=args.memory_budget_gib,
        backend="torch-eager" if args.device == "mps" else "torch", quantization="none",
        vae_core_frames=512 if args.device == "mps" else None,
        offload_ar=args.device == "mps",
    ) as pipeline:
        # Planning from an external score only tokenizes it; it performs no generation.
        plan = pipeline.plan(**request)
        budget = cover_token_budget(score_seconds, len(plan.prefix), pipeline.generation_config.context)
        (Path(args.output) / "generation_budget.json").write_text(json.dumps(budget, indent=2) + "\n")
        print(f"[length] score={score_seconds:.2f}s, semantic max_tokens={budget['semantic_max_tokens']} "
              f"(audio ceiling {budget['audio_ceiling_s']:.0f}s)", flush=True)
        if args.device == "cuda":
            result = pipeline(**request, semantic_sampling={"max_tokens": budget["semantic_max_tokens"]})
            result.save_artifacts(Path(args.output) / "generated")
            if any(result.truncated.values()):
                raise RuntimeError(f"YuE2 result truncated: {result.truncated}")
            return
        config = pipeline.effective_config(plan.request, semantic_sampling={"max_tokens": budget["semantic_max_tokens"]})
        request_id = identity({"request": plan.request.to_dict(), "config": config, "weights": pipeline.weights})
        # Keep costly intermediate results even if a later phase runs out of
        # memory. Reuse only matching, hash-checked artifacts from this run.
        from ..common.resources import upstream_code
        from ..runtime import runtime_identity
        checkpoint_key = checkpoint_identity({
            "request": request_id, "upstream": upstream_code(ROOT, "yue"),
            "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": runtime_identity(),
        })
        out = Path(args.output)
        plan.save(out / "plan")
        tokens, metadata = restore_cover_checkpoint(out, "semantic", checkpoint_key)
        if tokens is None:
            semantic = pipeline.generate_semantic(plan, sampling={"max_tokens": budget["semantic_max_tokens"]})
            if semantic.truncated:
                raise RuntimeError("YuE2 semantic generation truncated")
            tokens = np.asarray(semantic.tokens, dtype=np.int32)
            metadata = {"timing": semantic.timing}
            save_cover_checkpoint(out, "semantic", checkpoint_key, tokens, metadata)
        else:
            semantic = SemanticResult(plan, tokens.tolist(), metadata["timing"], False)
        # A retry can produce different tokens even with the same seed. The
        # acoustic result belongs to these exact tokens, not just the request.
        latent_key = cover_latent_key(checkpoint_key, tokens)
        latents, latent_metadata = restore_cover_checkpoint(out, "latent", latent_key)
        if latents is None:
            start = time.monotonic()
            latents = pipeline.synthesize(semantic)
            if not np.isfinite(latents).all():
                raise RuntimeError("YuE2 produced non-finite latents")
            latent_metadata = {"seconds": time.monotonic() - start}
            save_cover_checkpoint(out, "latent", latent_key, latents, latent_metadata)
        start = time.monotonic()
        audio = pipeline.decode(latents)
        timing = {"abc": plan.timing, "semantic": semantic.timing,
                  "nar_seconds": latent_metadata["seconds"], "vae_seconds": time.monotonic() - start,
                  "load": dict(pipeline.load_timing), "e2e_seconds": time.monotonic() - cover_start}
        result = SongResult(audio, 48000, semantic, latents, config, pipeline.weights, timing, request_id)
        result.save_artifacts(Path(args.output) / "generated")
        if any(result.truncated.values()):
            raise RuntimeError(f"YuE2 result truncated: {result.truncated}")


def cover_latent_key(request_key, tokens):
    return checkpoint_identity({"request": request_key,
                                "semantic_sha256": hashlib.sha256(tokens.tobytes()).hexdigest()})


def save_cover_checkpoint(output, phase, key, values, metadata):
    import numpy as np
    target = output / f"{phase}.npy"
    np.save(target, values, allow_pickle=False)
    write_json(output / f"{phase}.checkpoint.json", {
        "key": key, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "metadata": metadata})


def restore_cover_checkpoint(output, phase, key):
    import numpy as np
    candidates = [output]
    if output.name == "artifacts" and output.parent.name.startswith("attempt_"):
        candidates += sorted((p for p in output.parent.parent.glob("attempt_*/artifacts")
                              if p != output and not (p.parent / "outputs.json").exists()), reverse=True)
    for directory in candidates:
        marker, target = directory / f"{phase}.checkpoint.json", directory / f"{phase}.npy"
        if not marker.is_file() or not target.is_file():
            continue
        try:
            record = json.loads(marker.read_text())
            if not isinstance(record, dict) or record.get("key") != key:
                continue
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if phase == "semantic" and not isinstance(metadata.get("timing"), dict):
                continue
            if phase == "latent":
                seconds = metadata.get("seconds")
                if type(seconds) not in (int, float) or not 0 <= seconds < math.inf:
                    continue
            # The copied marker must be writable as strict JSON, including nested timing.
            json.dumps(metadata, allow_nan=False)
            if hashlib.sha256(target.read_bytes()).hexdigest() != record.get("sha256"):
                continue
            with target.open("rb") as array:
                values = np.lib.format.read_array(array, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            continue
        if not values.size:
            continue
        if phase == "semantic" and (values.ndim != 1 or values.dtype != np.int32):
            continue
        if phase == "latent":
            # YuE2 VAE accepts [T, 64] or [1, 64, T].
            valid_shape = values.shape[1:] == (64,) or (values.ndim == 3 and values.shape[:2] == (1, 64))
            if not valid_shape or values.dtype.kind != "f" or not np.isfinite(values).all():
                continue
        if directory != output:
            save_cover_checkpoint(output, phase, key, values, metadata)
        print(f"[reuse] YuE2 {phase}: {directory}", flush=True)
        return values, metadata
    return None, None
