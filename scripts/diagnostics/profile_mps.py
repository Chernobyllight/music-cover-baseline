#!/usr/bin/env python3
"""Bounded, isolated diagnostics using artifacts from a successful run.

Each GPU worker has a 60-second process deadline. The shared ledger charges
the entire worker lifetime (including imports/loading) to its authorized
allowance and enforces five minutes between workers across all rounds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def write_json(path, value):
    # A killed process must leave the last completed report or reservation readable.
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    pending.replace(path)


def artifact(run, stage, key):
    state = json.loads((run / "stage_state.json").read_text())
    entry = state["stages"][stage]
    if entry["status"] != "succeeded":
        raise ValueError(f"Baseline stage {stage} was not successful")
    path = Path(entry["outputs"][key]["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def measure(fn):
    import torch
    torch.mps.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.mps.synchronize()
    return value, time.perf_counter() - start


def nar_probe(run, output):
    import numpy as np
    import torch
    from yue2.nar import attention
    from yue2.protocol import CODEC_SIZE
    generated = artifact(run, "02_cover", "cover").parent
    prefix = np.load(generated / "prefix.npy")
    tokens = np.load(generated / "semantic.npy")
    frames = int(np.sum((tokens >= 0) & (tokens < CODEC_SIZE)))
    config = json.loads((ROOT / "models/YuE2-3B/config.json").read_text())
    qlen, klen = frames + 2, len(prefix) + frames + 1 + frames + 2
    generator = torch.Generator().manual_seed(42)
    def tensor(length, heads):
        return torch.randn(length, heads, config["head_dim"], generator=generator).to(device="mps", dtype=torch.bfloat16)
    q = tensor(qlen, config["num_attention_heads"])
    k, v = (tensor(klen, config["num_key_value_heads"]) for _ in range(2))
    report = {"kind": "synthetic_activations_real_song_shapes", "q_shape": list(q.shape),
              "k_shape": list(k.shape), "dtype": str(q.dtype), "variants": {}}
    baseline = None
    for block in (256, 512, 1024):
        _, warm = measure(lambda: attention(q, k, v, query_chunk_size=block))
        actual, seconds = measure(lambda: attention(q, k, v, query_chunk_size=block))
        actual = actual.cpu()
        if baseline is None:
            baseline = actual
        report["variants"][str(block)] = {"warmup_seconds": warm, "seconds": seconds,
            "equal_to_256": torch.equal(actual, baseline),
            "max_abs_diff": float((actual.float() - baseline.float()).abs().max()),
            "finite": bool(torch.isfinite(actual).all())}
        write_json(output / "probe.json", report)
    _, seconds = measure(lambda: (k.transpose(0, 1).unsqueeze(0).repeat_interleave(2, 1),
                                  v.transpose(0, 1).unsqueeze(0).repeat_interleave(2, 1)))
    report["kv_expansion_seconds"] = seconds
    write_json(output / "probe.json", report)


def ar_probe(run, output, reference_sdpa=None):
    import numpy as np
    import torch
    from contextlib import nullcontext
    from music_cover.runtime import ForwardAudit
    from yue2.modeling_yue2 import YuE2ForCausalLM, StaticKVCache
    from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, Sampling
    from yue2.sampling import distribution
    generated = artifact(run, "02_cover", "cover").parent
    prefix = np.load(generated / "prefix.npy").tolist()
    codec = np.load(generated / "semantic.npy").tolist()
    codec = [int(t) + CODEC_OFFSET for t in codec if 0 <= t < CODEC_SIZE]
    report = {"kind": "real_weights_fixed_token_replay", "contexts": {}, "sampling": {}}
    start = time.perf_counter()
    model = YuE2ForCausalLM.from_pretrained(ROOT / "models/YuE2-3B", local_files_only=True,
                  torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to("mps")
    torch.mps.synchronize()
    report["load_seconds"] = time.perf_counter() - start
    config = model.config
    for count in (0, len(codec) - 16):
        ids = prefix + codec[:count]
        cache = StaticKVCache(config.num_hidden_layers, 1, config.num_key_value_heads,
                              len(ids) + 16, config.head_dim, torch.bfloat16, torch.device("mps"))
        def prefill():
            result = None
            for start in range(0, len(ids), 1024):
                result = model(torch.tensor([ids[start:start + 1024]], device="mps"),
                    past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
            return result
        logits, prefill_seconds = measure(prefill)
        np.save(output / f"logits-{len(ids)}.npy", logits.float().cpu().numpy())
        tokens = [torch.tensor([[t]], device="mps") for t in codec[count:count + 8]]
        position = cache._seen_tokens
        def replay():
            cache._seen_tokens = position
            result = None
            for token in tokens:
                result = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
            return result
        # Warm identical positions before the single A/B comparison.
        _, warm_seconds = measure(replay)
        with ForwardAudit("mps"):
            audited, on = measure(replay)
        plain, off = measure(replay)
        np.save(output / f"replay-logits-{len(ids)}.npy", plain.float().cpu().numpy())
        report["contexts"][str(len(ids))] = {"tokens": len(tokens), "prefill_seconds": prefill_seconds,
            "warmup_seconds": warm_seconds, "audit_on_seconds": on, "audit_off_seconds": off,
            "equal": torch.equal(audited.cpu(), plain.cpu()), "finite": bool(torch.isfinite(plain).all().cpu())}
        if reference_sdpa is not None:
            from unittest.mock import patch
            with patch("yue2.modeling_yue2.sdpa", reference_sdpa):
                reference, reference_seconds = measure(replay)
            expected, actual = reference.float().cpu(), plain.float().cpu()
            report["contexts"][str(len(ids))]["expanded_reference"] = {
                "seconds": reference_seconds, "equal": torch.equal(expected, actual),
                "max_abs_diff": float((expected - actual).abs().max())}
            write_json(output / "probe.json", report)
            if not torch.equal(expected, actual):
                raise FloatingPointError("Patched AR fixed-token replay differs from expanded K/V")
        write_json(output / "probe.json", report)
        history = codec[max(0, count - 50):count]
        sampling = Sampling()
        scores, dist_time = measure(lambda: distribution(logits, sampling, history, max(200, count), "semantic"))
        probabilities, softmax_time = measure(lambda: scores.softmax(-1))
        cpu_prob, transfer_time = measure(lambda: probabilities.cpu())
        rng = torch.Generator().manual_seed(42)
        start = time.perf_counter()
        chosen = torch.multinomial(cpu_prob, 1, generator=rng)
        multinomial_time = time.perf_counter() - start
        _, roundtrip = measure(lambda: int(chosen.to("mps").item()))
        # CPU score processing is diagnostic only; never changes production sampling.
        cpu_logits = logits.cpu()
        start = time.perf_counter()
        cpu_scores = distribution(cpu_logits, sampling, history, max(200, count), "semantic")
        cpu_distribution_time = time.perf_counter() - start
        report["sampling"][str(len(ids))] = {"distribution_seconds": dist_time,
            "softmax_seconds": softmax_time, "probabilities_to_cpu_seconds": transfer_time,
            "cpu_multinomial_seconds": multinomial_time, "token_gpu_roundtrip_seconds": roundtrip,
            "cpu_distribution_seconds": cpu_distribution_time,
            "cpu_scores_equal": torch.equal(cpu_scores, scores.cpu())}
        del cache, audited, plain
        torch.mps.empty_cache()
        write_json(output / "probe.json", report)


def verify_yue_probe(run, output):
    """Exercise the applied candidate against the actual pinned SDPA function."""
    import ast
    import torch
    import torch.nn.functional as F
    import unittest
    from yue2 import modeling_yue2
    if not getattr(modeling_yue2, "_MPS_NATIVE_GQA", False):
        raise RuntimeError("Apply the native AR candidate patch and use PyTorch >= 2.15 before verification")
    source = subprocess.check_output(["git", "-C", str(ROOT / "third_party/YuE"), "show",
        "1647252d68b70cbead046ffd3f7caf036dde47ee:src/yue2/modeling_yue2.py"], text=True)
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "sdpa")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned_sdpa", "exec"), namespace)
    ar_probe(run, output, reference_sdpa=namespace["sdpa"])
    sys.path.insert(0, str(ROOT / "tests"))
    import test_yue_attention
    os.environ["MUSIC_TEST_DEVICE"] = "mps"
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_yue_attention))
    write_json(output / "attention_regression.json", {"tests": result.testsRun, "passed": result.wasSuccessful()})
    if not result.wasSuccessful():
        raise RuntimeError("Patched AR MPS attention regression failed")


def upgrade_yue_probe(run, output):
    """Separate dependency-only behavior from native SDPA call candidates."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from yue2.nar import attention
    from yue2 import modeling_yue2
    from yue2.protocol import CODEC_SIZE
    if getattr(modeling_yue2, "_MPS_NATIVE_GQA", False):
        raise RuntimeError("Dependency-only measurement requires reversing the native AR candidate patch")
    generated = artifact(run, "02_cover", "cover").parent
    prefix = np.load(generated / "prefix.npy")
    tokens = np.load(generated / "semantic.npy")
    frames = int(np.sum((tokens >= 0) & (tokens < CODEC_SIZE)))
    g = torch.Generator().manual_seed(42)
    q = torch.randn(frames + 2, 16, 128, generator=g).bfloat16()
    k = torch.randn(len(prefix) + frames * 2 + 3, 8, 128, generator=g).bfloat16()
    v = torch.randn(k.shape, generator=g).bfloat16()
    rows = [0, 1, 255, 256, 511, 512, 1024, frames + 1]
    reference = F.scaled_dot_product_attention(q[rows].float().transpose(0, 1)[None],
        k.float().transpose(0, 1)[None], v.float().transpose(0, 1)[None], enable_gqa=True)[0].transpose(0, 1)
    q, k, v = (x.to("mps") for x in (q, k, v))
    def native(block=None):
        query, key, value = (x.transpose(0, 1)[None] for x in (q, k, v))
        block = block or len(q)
        parts = [F.scaled_dot_product_attention(query[..., i:i + block, :], key, value, enable_gqa=True)
                 for i in range(0, len(q), block)]
        return torch.cat(parts, dim=-2)[0].transpose(0, 1)
    report = {"torch": torch.__version__, "kind": "dependency_and_callsite_separate_candidates",
              "reference": "CPU FP32 on identical quantized inputs", "atol": 0.001, "rtol": 0.02,
              "variants": {}, "edge_cases": []}
    for name, fn in (("unchanged_callsite_256", lambda: attention(q, k, v)),
                     ("native_gqa_256", lambda: native(256)), ("native_gqa_full", native)):
        _, warm = measure(fn)
        result, seconds = measure(fn)
        actual = result.cpu()[rows].float()
        report["variants"][name] = {"warmup_seconds": warm, "seconds": seconds,
            "finite": bool(torch.isfinite(result).all().cpu()),
            "max_abs_error": float((actual-reference).abs().max()),
            "reference_passed": bool(torch.allclose(actual, reference, atol=0.001, rtol=0.02))}
        write_json(output / "attention.json", report)
    # Small-tail, noncontiguous, causal and explicitly masked rectangular inputs.
    for qlen, klen, causal, masked in [(n, 257, False, False) for n in range(1, 9)] + [(33, 33, True, False), (7, 257, False, True)]:
        cq = torch.randn(qlen, 16, 128, generator=g).bfloat16().transpose(0, 1)[None]
        ck = torch.randn(klen, 8, 128, generator=g).bfloat16().transpose(0, 1)[None]
        cv = torch.randn(klen, 8, 128, generator=g).bfloat16().transpose(0, 1)[None]
        mask = torch.arange(klen)[None] <= torch.arange(klen-qlen, klen)[:, None] if masked else None
        expected = F.scaled_dot_product_attention(cq.float(), ck.float(), cv.float(),
            is_causal=causal, attn_mask=mask, enable_gqa=True)
        actual, _ = measure(lambda: F.scaled_dot_product_attention(cq.to("mps"), ck.to("mps"), cv.to("mps"),
            is_causal=causal, attn_mask=mask.to("mps") if mask is not None else None, enable_gqa=True))
        actual = actual.float().cpu()
        report["edge_cases"].append({"qlen":qlen, "klen":klen, "causal":causal, "masked":masked,
            "max_abs_error":float((actual-expected).abs().max()),
            "reference_passed":bool(torch.allclose(actual, expected, atol=0.001, rtol=0.02))})
        write_json(output / "attention.json", report)
    if not all(x["reference_passed"] and x["finite"] for x in report["variants"].values()) or not all(
            x["reference_passed"] for x in report["edge_cases"]):
        raise FloatingPointError("Native attention failed the CPU FP32 reference; see attention.json")
    ar_output = output / "ar"
    ar_output.mkdir()
    ar_probe(run, ar_output)


def native_yue_probe(run, output):
    """Diagnostic callsite substitution only; never changes installed sources."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from unittest.mock import patch
    from music_cover.runtime import ForwardAudit
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.nar import CachedNAR, song_chunks
    from yue2.protocol import CODEC_SIZE
    def native_ar(query, key, value, *, attn_mask=None, is_causal=False):
        return F.scaled_dot_product_attention(query, key, value,
            attn_mask=attn_mask, is_causal=is_causal, enable_gqa=query.shape[1] != key.shape[1])
    def native_nar(q, k, v, *, causal=False, backend="sdpa", query_chunk_size=None):
        if backend != "sdpa" or query_chunk_size is not None:
            raise ValueError("This diagnostic measures the default SDPA call only")
        return native_ar(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                         v.transpose(0, 1)[None], is_causal=causal)[0].transpose(0, 1)
    ar_output = output / "ar"
    ar_output.mkdir()
    with patch("yue2.modeling_yue2.sdpa", native_ar):
        ar_probe(run, ar_output)
    torch.mps.empty_cache()
    generated = artifact(run, "02_cover", "cover").parent
    prefix = np.load(generated / "prefix.npy").tolist()
    codec = [int(t) for t in np.load(generated / "semantic.npy") if 0 <= t < CODEC_SIZE]
    chunk = song_chunks(prefix, codec, 42)[0]
    model = YuE2ForCausalLM.from_pretrained(ROOT / "models/YuE2-3B", local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to("mps")
    from contextlib import nullcontext
    report = {"kind": "real_full_shape_single_NAR_velocity_not_32_step_synthesis", "variants": {}}
    values = {}
    for name, context in (("unchanged", nullcontext()), ("native", patch("yue2.nar.attention", native_nar))):
        with context, ForwardAudit("mps"):
            cached, prefill_seconds = measure(lambda: CachedNAR(model, chunk))
            state = chunk.noise.to(device="mps", dtype=torch.bfloat16)
            value, seconds = measure(lambda: cached.velocity(state, 0.0))
            values[name] = value.float().cpu()
            np.save(output / f"nar-{name}.npy", values[name].numpy())
            report["variants"][name] = {"prefill_seconds": prefill_seconds, "first_velocity_seconds": seconds,
                "finite": bool(torch.isfinite(values[name]).all()), "shape": list(value.shape)}
            cached.close()
            del cached, state, value
            torch.mps.empty_cache()
        write_json(output / "nar.json", report)
    difference = values["native"] - values["unchanged"]
    report["comparison"] = {"max_abs_diff": float(difference.abs().max()),
        "relative_rms": float(difference.square().mean().sqrt() / values["unchanged"].square().mean().sqrt()),
        "cosine": float(F.cosine_similarity(values["native"].flatten(), values["unchanged"].flatten(), dim=0))}
    write_json(output / "nar.json", report)


def layout_probe(run, output):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from unittest.mock import patch
    from yue2.nar import attention
    from yue2.protocol import CODEC_SIZE
    generated = artifact(run, "02_cover", "cover").parent
    prefix = np.load(generated / "prefix.npy")
    tokens = np.load(generated / "semantic.npy")
    frames = int(np.sum((tokens >= 0) & (tokens < CODEC_SIZE)))
    generator = torch.Generator().manual_seed(42)
    q = torch.randn(frames + 2, 16, 128, generator=generator).bfloat16()
    k = torch.randn(len(prefix) + frames * 2 + 3, 8, 128, generator=generator).bfloat16()
    v = torch.randn(k.shape, generator=generator).bfloat16()
    rows = [0, 1, 255, 256, 511, 512, 1024, frames + 1]
    reference = F.scaled_dot_product_attention(q[rows].float().transpose(0, 1)[None],
        k.float().transpose(0, 1)[None].repeat_interleave(2, 1),
        v.float().transpose(0, 1)[None].repeat_interleave(2, 1))[0].transpose(0, 1)
    q, k, v = (x.to("mps") for x in (q, k, v))
    original = F.scaled_dot_product_attention
    report = {"kind": "layout_diagnostic_not_production_change", "reference_rows": rows, "variants": {}}
    for name, block, contiguous in (("original_256", 256, False), ("contiguous_256", 256, True), ("contiguous_512", 512, True)):
        def sdpa(q, k, v, **kwargs):
            if contiguous:
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            return original(q, k, v, **kwargs)
        with patch("torch.nn.functional.scaled_dot_product_attention", side_effect=sdpa):
            _, warm = measure(lambda: attention(q, k, v, query_chunk_size=block))
            actual, seconds = measure(lambda: attention(q, k, v, query_chunk_size=block))
        diff = (actual.cpu()[rows].float() - reference).abs()
        report["variants"][name] = {"seconds": seconds, "warmup_seconds": warm,
            "cpu_reference_max_abs_diff": float(diff.max()),
            "cpu_reference_row_max_abs_diff": diff.flatten(1).amax(1).tolist()}
        write_json(output / "probe.json", report)
    bridge_probe(output)
    # Tiny real MPS model tests, not release-checkpoint/audio generation.
    import unittest
    sys.path.insert(0, str(ROOT / "tests"))
    import test_yue_sampling
    os.environ["MUSIC_TEST_DEVICE"] = "mps"
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_yue_sampling)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(output / "sampling_regression.json", {"device": "mps", "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful()})
    if not result.wasSuccessful():
        raise RuntimeError("MPS sampling regression failed")


def bridge_probe(output):
    import torch
    # Exact same CPU RNG and probabilities; only remove the redundant GPU
    # scalar readback after sampling. This is independent of model precision.
    probabilities = torch.zeros(1, 184704)
    probabilities[0, 151853:151953] = 0.01
    bridge = {}
    choices = {}
    for name in ("gpu_readback", "cpu_scalar"):
        rng = torch.Generator().manual_seed(42)
        def sample():
            tokens = []
            next_id = None
            for _ in range(32):
                chosen = torch.multinomial(probabilities, 1, generator=rng)
                if name == "gpu_readback":
                    next_id = chosen.to("mps")
                    tokens.append(int(next_id.item()))
                else:
                    tokens.append(int(chosen.item()))
                    next_id = chosen.to("mps")
            return tokens, next_id
        _, warm = measure(sample)
        (tokens, next_id), seconds = measure(sample)
        choices[name] = (tokens, rng.get_state())
        bridge[name] = {"seconds": seconds, "warmup_seconds": warm, "tokens": len(tokens), "next_input_device": str(next_id.device)}
    bridge["identical_tokens_and_rng"] = (choices["gpu_readback"][0] == choices["cpu_scalar"][0]
        and torch.equal(choices["gpu_readback"][1], choices["cpu_scalar"][1]))
    write_json(output / "sampling_bridge.json", bridge)


def seed_probe(run, output, compare_fp16=False):
    import soundfile as sf
    import torch
    torch.manual_seed(42)
    project = ROOT / "third_party/seed-vc"
    os.chdir(project)
    sys.path.insert(0, str(project))
    import inference
    inference.device = torch.device("mps")
    assets = {("lj1995/VoiceConversionWebUI", "rmvpe.pt"): ROOT / "models/RMVPE/rmvpe.pt",
              ("funasr/campplus", "campplus_cn_common.bin"): ROOT / "models/CAMPPlus/campplus_cn_common.bin"}
    inference.load_custom_model_from_hf = lambda repo, model_filename, config_filename=None: str(assets[repo, model_filename])
    source, rate = sf.read(artifact(run, "04_convert", "converted").parent / "source_mono.wav", frames=44100 * 20, dtype="float32")
    sf.write(output / "source-20s.wav", source, rate, subtype="FLOAT")
    options = argparse.Namespace(source=str(output / "source-20s.wav"),
        target=str(artifact(run, "00_prepare", "reference")), output=str(output),
        diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=0.7, f0_condition=True,
        auto_f0_adjust=False, semi_tone_shift=0, fp16=False,
        checkpoint=str(ROOT / "models/Seed-VC/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"),
        config=str(ROOT / "configs/seed_svc.local.yml"))
    report = {"kind": "real_20s_features_one_diffusion_forward_and_source_mel_vocoder",
              "calls": {}, "production_steps": 30, "diagnostic_estimator_calls": 0}
    original_load = inference.load_models
    captured = {}
    class ForwardComplete(Exception):
        pass
    def timed(name, fn, save=False):
        def call(*args, **kwargs):
            result, seconds = measure(lambda: fn(*args, **kwargs))
            report["calls"].setdefault(name, []).append(seconds)
            if save and name not in captured:
                captured[name] = result
            write_json(output / "probe.json", report)
            return result
        return call
    def load(args):
        parts, seconds = measure(lambda: original_load(args))
        report["load_seconds"] = seconds
        model, semantic, f0, vocoder, campplus, mel, mel_args = parts
        captured["vocoder"] = vocoder
        original_estimator = model.cfm.estimator.forward
        def first_step(*args, **kwargs):
            value, seconds = measure(lambda: original_estimator(*args, **kwargs))
            report["diagnostic_estimator_calls"] += 1
            report["calls"]["dit_first_forward"] = [seconds]
            report["dit_input_shape"] = list(args[0].shape)
            report["dit_dtype"] = str(args[0].dtype)
            report["dit_input_device"] = str(args[0].device)
            report["dit_output_device"] = str(value.device)
            report["dit_parameter_device"] = str(next(model.cfm.estimator.parameters()).device)
            report["dit_finite"] = bool(torch.isfinite(value).all().cpu())
            # Persist a CPU copy so the upgraded runtime can replay identical
            # diffusion inputs, independently of feature-extractor differences.
            def cpu(x):
                return x.detach().cpu() if isinstance(x, torch.Tensor) else x
            torch.save({"args": tuple(cpu(x) for x in args),
                        "kwargs": {k: cpu(v) for k, v in kwargs.items()},
                        "output": value.cpu()}, output / "dit-inputs.pt")
            if compare_fp16:
                references = list(output.parent.glob("*_seed/dit-inputs.pt"))
                if len(references) != 1:
                    raise ValueError("Expected one saved baseline Seed-VC forward in the shared ledger")
                saved = torch.load(references[0], map_location="cpu", weights_only=True)
                def device(x):
                    return x.to("mps") if isinstance(x, torch.Tensor) else x
                fixed_args = tuple(device(x) for x in saved["args"])
                fixed_kwargs = {k: device(v) for k, v in saved["kwargs"].items()}
                baseline = saved["output"].float()
                report["fixed_input_reference"] = str(references[0])
                report["precision_candidates"] = {}
                new_fp32 = None
                for name, enabled in (("fp32", False), ("fp16", True)):
                    with torch.autocast("mps", dtype=torch.float16, enabled=enabled):
                        _, warm = measure(lambda: original_estimator(*fixed_args, **fixed_kwargs))
                        report["diagnostic_estimator_calls"] += 1
                        actual, elapsed = measure(lambda: original_estimator(*fixed_args, **fixed_kwargs))
                        report["diagnostic_estimator_calls"] += 1
                    actual = actual.float().cpu()
                    torch.save(actual, output / f"dit-{name}.pt")
                    reference = baseline if name == "fp32" else new_fp32
                    diff = actual - reference
                    report["precision_candidates"][name] = {"warmup_seconds": warm, "seconds": elapsed,
                        "reference": "old_runtime_fp32" if name == "fp32" else "new_runtime_fp32",
                        "finite": bool(torch.isfinite(actual).all()), "max_abs_diff": float(diff.abs().max()),
                        "relative_rms": float(diff.square().mean().sqrt() / reference.square().mean().sqrt()),
                        "cosine": float(torch.nn.functional.cosine_similarity(actual.flatten(), reference.flatten(), dim=0))}
                    if name == "fp32":
                        new_fp32 = actual
                    write_json(output / "probe.json", report)
            write_json(output / "probe.json", report)
            raise ForwardComplete()
        model.cfm.estimator.forward = first_step
        model.length_regulator.forward = timed("length_regulator", model.length_regulator.forward)
        campplus.forward = timed("campplus", campplus.forward)
        return (model, timed("whisper", semantic), timed("rmvpe", f0), vocoder,
                campplus, timed("mel", mel, save=True), mel_args)
    inference.load_models = load
    try:
        inference.main(options)
        raise RuntimeError("Diagnostic stop was not reached")
    except ForwardComplete:
        pass
    # The real source mel keeps this probe independent of diffusion quality.
    audio, seconds = measure(lambda: captured["vocoder"](captured["mel"].float()))
    report["calls"]["bigvgan_source_mel"] = [seconds]
    report["vocoder_finite"] = bool(torch.isfinite(audio).all().cpu())
    write_json(output / "probe.json", report)
    if not report["dit_finite"] or not report["vocoder_finite"] or not all(
            item["finite"] for item in report.get("precision_candidates", {}).values()):
        raise FloatingPointError("Seed-VC diagnostic produced non-finite output; see probe.json")


def upgrade_seed_probe(run, output):
    seed_probe(run, output, compare_fp16=True)


def worker(args):
    import torch
    from music_cover.runtime import MemoryMonitor, probe, resolve_device, source_identity
    resolve_device("mps")
    torch.set_num_threads(4)
    with torch.inference_mode(), MemoryMonitor("mps", 16) as monitor:
        try:
            environment = probe("mps")
            environment["upstream"] = source_identity(ROOT)
            environment["interpreter"] = sys.executable
            write_json(args.output / "environment.json", environment)
            if args.mode == "seed-phrase":
                from profile_seed_phrase import run_phrase
                run_phrase(args.run, args.output)
                return
            if args.mode == "yue-head":
                from profile_yue_head import run_head
                run_head(args.run, args.output)
                return
            if args.mode in ROUND4:
                import profile_mps_round4 as exp
                getattr(exp, args.mode.replace("-", "_"))(args.run, args.output)
                return
            if args.mode == "separator":
                from profile_separator_block import run_block
                run_block(args.run, args.output)
                return
            {"nar": nar_probe, "ar": ar_probe, "seed": seed_probe, "layout": layout_probe,
             "upgrade-yue": upgrade_yue_probe, "native-yue": native_yue_probe,
             "upgrade-seed": upgrade_seed_probe, "verify-yue": verify_yue_probe}[args.mode](args.run, args.output)
        finally:
            write_json(args.output / "memory.json", monitor.report())


ROUND4 = {"nar-cost", "nar-fuse", "seed-cost", "seed-combo", "ar-combo"}
GRANT = "m5-ten-minute-round"


def round_allowance(ledger, mode):
    if mode not in ROUND4:
        used = sum(w["charged_seconds"] for w in ledger["workers"] if "grant" not in w)
        return 300 - used
    grants = ledger.setdefault("grants", {})
    grants.setdefault(GRANT, {"seconds": 300, "authorized_for": sorted(ROUND4)})
    used = sum(w["charged_seconds"] for w in ledger["workers"] if w.get("grant") == GRANT)
    return grants[GRANT]["seconds"] - used


def launch(args):
    import fcntl
    import hashlib
    args.output.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output / "budget.json"
    with (args.output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode in ROUND4:
            canonical = ROOT / "runs/probes/mps-performance"
            if args.output.resolve() != canonical.resolve() or not ledger_path.is_file():
                raise SystemExit("Use the original shared MPS ledger with its existing history")
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"workers": []}
        now = time.time()
        if ledger["workers"] and now - ledger["workers"][-1]["ended_at"] < 300:
            raise SystemExit("Cooling interval has not reached 300 seconds; no worker started")
        cap = {"seed-phrase": 38, "yue-head": 11}.get(args.mode, 60)
        limit = min(getattr(args, "deadline", 60), cap)
        if not 1 <= limit <= 60:
            raise SystemExit("Deadline must be between 1 and 60 seconds")
        if limit > round_allowance(ledger, args.mode):
            raise SystemExit("Not enough diagnostic budget for a bounded worker")
        destination = args.output / f"{len(ledger['workers']) + 1:02d}_{args.mode}"
        destination.mkdir()
        environment = "seed-vc" if args.mode in {"seed", "upgrade-seed"} else "yue2"
        if args.mode == "separator":
            environment = "separator"
        if args.mode in {"seed-phrase", "yue-head"}:
            environment = "seed-vc-torch215" if args.mode == "seed-phrase" else "yue2-torch215"
        if args.mode in ROUND4:
            environment = "seed-vc-torch215" if args.mode.startswith("seed") else "yue2-torch215"
        interpreter = getattr(args, "python", None) or ROOT / f"envs/{environment}/bin/python"
        command = [str(Path(interpreter).absolute()), str(Path(__file__).resolve()),
                   args.mode, "--run", str(args.run), "--output", str(destination), "--worker"]
        env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0", "HF_HUB_OFFLINE": "1",
               "HF_HOME": str(ROOT / "cache/huggingface")}
        start = time.monotonic()
        entry = {"mode": args.mode, "started_at": now, "ended_at": now + limit,
                 "deadline_seconds": limit, "charged_seconds": limit, "status": "reserved", "command": command,
                 "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "thermal_before": subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout}
        if args.mode == "separator":
            helper = Path(__file__).with_name("profile_separator_block.py")
            entry["block_script_sha256"] = hashlib.sha256(helper.read_bytes()).hexdigest()
        if args.mode in {"seed-phrase", "yue-head"}:
            helper = Path(__file__).with_name("profile_" + args.mode.replace("-", "_") + ".py")
            entry["helper_sha256"] = hashlib.sha256(helper.read_bytes()).hexdigest()
        if args.mode in ROUND4:
            entry["grant"] = GRANT
            if args.mode.startswith("seed"):
                dependencies = [Path(__file__).with_name("profile_seed_phrase.py"),
                                *(ROOT / f"patches/candidates/seed-vc-{name}.patch"
                                  for name in ("key-mask", "equal-heads", "loop"))]
                entry["diagnostic_dependencies"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in dependencies}
            entry["helper_sha256"] = hashlib.sha256(Path(__file__).with_name("profile_mps_round4.py").read_bytes()).hexdigest()
        ledger["workers"].append(entry)
        write_json(ledger_path, ledger)
        try:
            with (destination / "process.log").open("w") as log:
                env["MUSIC_PROBE_END"] = str(start + limit)
                result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=max(0, start + limit - time.monotonic()))
                entry["status"] = "passed" if result.returncode == 0 else "failed"
                entry["exit_code"] = result.returncode
        except subprocess.TimeoutExpired:
            entry["status"] = "deadline_exceeded"
        finally:
            entry["ended_at"] = time.time()
            entry["charged_seconds"] = time.monotonic() - start
            entry["thermal_after"] = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
            write_json(ledger_path, ledger)
        print(json.dumps(entry, ensure_ascii=False))
        if entry["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["nar", "ar", "seed", "layout", "upgrade-yue", "native-yue", "upgrade-seed", "verify-yue", "separator", "seed-phrase", "yue-head", *sorted(ROUND4)])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Shared diagnostic ledger directory")
    parser.add_argument("--python", type=Path, help="Candidate interpreter; uses the same budget and cooldown ledger")
    parser.add_argument("--deadline", type=int, default=60, help="Worker wall-clock limit in seconds (1–60)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    worker(args) if args.worker else launch(args)
