from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
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
    from yue2 import YuE2Pipeline
    sys.path.insert(0, str(ROOT / "src"))
    from music_cover.pitch import abc_tools
    request = json.loads(Path(args.request).read_text())
    request["abc"] = Path(args.score).read_text()
    request["cot"] = "melody"
    score = abc_tools().parse(request["abc"])
    score_seconds = float(score.voices["Vocal"].time * 60 / score.bpm)
    with YuE2Pipeline.from_pretrained(
        str(ROOT / "models/YuE2-3B"), vae=str(ROOT / "models/YuE2-Vae"),
        device="cuda", local_files_only=True, memory_budget_gib=80,
    ) as pipeline:
        # Planning from an external score only tokenizes it; it performs no generation.
        plan = pipeline.plan(**request)
        budget = cover_token_budget(score_seconds, len(plan.prefix), pipeline.generation_config.context)
        (Path(args.output) / "generation_budget.json").write_text(json.dumps(budget, indent=2) + "\n")
        print(f"[length] score={score_seconds:.2f}s, semantic max_tokens={budget['semantic_max_tokens']} "
              f"(audio ceiling {budget['audio_ceiling_s']:.0f}s)", flush=True)
        result = pipeline(**request, semantic_sampling={"max_tokens": budget["semantic_max_tokens"]})
        result.save_artifacts(Path(args.output) / "generated")
        if any(result.truncated.values()):
            raise RuntimeError(f"YuE2 result truncated: {result.truncated}")
