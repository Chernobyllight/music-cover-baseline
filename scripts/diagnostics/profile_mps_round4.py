"""Round four diagnostics. All accelerator calls run inside the shared launcher."""

from contextlib import ExitStack
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from profile_mps import ROOT, artifact, measure, write_json


def yue_inputs(run):
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.protocol import CODEC_SIZE

    folder = artifact(run, "02_cover", "cover").parent
    prefix = np.load(folder / "prefix.npy").tolist()
    codec = [int(t) for t in np.load(folder / "semantic.npy") if 0 <= t < CODEC_SIZE]
    model = (
        YuE2ForCausalLM.from_pretrained(
            ROOT / "models/YuE2-3B",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to("mps")
    )
    return model, prefix, codec


def nar_setup(run):
    from yue2.nar import CachedNAR, song_chunks

    model, prefix, codec = yue_inputs(run)
    chunk = song_chunks(prefix, codec, 42)[0]
    engine = CachedNAR(model, chunk)
    state = chunk.noise.to(device="mps", dtype=torch.bfloat16)
    return model, engine, state


def nar_cost(run, output):
    model, engine, state = nar_setup(run)
    layer_index = len(model.model.layers) // 2
    layer = model.model.layers[layer_index]
    saved = {}

    def capture(name):
        def hook(module, args):
            saved[name] = args[0]

        return hook

    for name, mod in [
        ("qkv", layer.nar_self_attn.q_proj),
        ("out", layer.nar_self_attn.o_proj),
        ("mlp", layer.nar_mlp),
    ]:
        saved[name + "_hook"] = mod.register_forward_pre_hook(capture(name))
    _, cold = measure(lambda: engine.velocity(state, 0.0))
    for name in ("qkv", "out", "mlp"):
        saved.pop(name + "_hook").remove()
    value, warm = measure(lambda: engine.velocity(state, 0.0))
    assert torch.isfinite(value).all().item()
    ar_k, ar_v = engine.cache[layer_index]
    qkv = lambda: layer.nar_self_attn.project_qkv(saved["qkv"], engine.cos, engine.sin)
    q, k, v = qkv()
    cat = lambda: (torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0])))
    keys, values = cat()
    operations = {
        "project_qkv_norm_rope": qkv,
        "cat_kv": cat,
        "attention": lambda: engine._attention(q[0], keys, values),
        "out_proj": lambda: layer.nar_self_attn.o_proj(saved["out"]),
        "mlp": lambda: layer.nar_mlp(saved["mlp"]),
    }
    times = {}
    for name, fn in operations.items():
        measure(fn)
        _, times[name] = measure(fn)
    write_json(
        output / "result.json",
        dict(
            layer=layer_index,
            layers=len(model.model.layers),
            cold_s=cold,
            velocity_s=warm,
            layer_seconds=times,
            state=list(state.shape),
        ),
    )
    engine.close()


def difference(value, reference):
    a, b = value.float().cpu(), reference.float().cpu()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError("Non-finite model output")
    return dict(
        max_abs=float((a - b).abs().max()),
        relative_rms=float((a - b).norm() / b.norm().clamp_min(1.0e-15)),
        exact=torch.equal(a, b),
    )


def nar_fuse(run, output):
    from yue2.modeling_yue2 import _apply_rotary

    model, engine, state = nar_setup(run)
    layers = model.model.layers
    reference, first = measure(lambda: engine.velocity(state, 0.0))
    _, warm = measure(lambda: engine.velocity(state, 0.0))
    report = dict(baseline_first_s=first, baseline_warm_s=warm, candidates={})

    def fused_attention(mod, weight):
        sizes = [m.out_features for m in (mod.q_proj, mod.k_proj, mod.v_proj)]

        def call(x, cos, sin):
            q, k, v = F.linear(x, weight).split(sizes, dim=-1)
            q, k, v = [item.unflatten(-1, (-1, mod.head_dim)) for item in (q, k, v)]
            cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
            return _apply_rotary(mod.q_norm(q), cos, sin), _apply_rotary(mod.k_norm(k), cos, sin), v

        return call

    def fused_mlp(mod, weight):
        def call(x):
            gate, up = F.linear(x, weight).chunk(2, dim=-1)
            return mod.down_proj(F.silu(gate) * up)

        return call

    for name in ("qkv", "gate_up"):
        packed_bytes = 0
        before = torch.mps.current_allocated_memory()
        with ExitStack() as stack:

            def install():
                nonlocal packed_bytes
                for layer in layers:
                    if name == "qkv":
                        mod = layer.nar_self_attn
                        parts = [mod.q_proj, mod.k_proj, mod.v_proj]
                        method, make = "project_qkv", fused_attention
                    else:
                        mod = layer.nar_mlp
                        parts = [mod.gate_proj, mod.up_proj]
                        method, make = "forward", fused_mlp
                    weight = torch.cat([part.weight for part in parts])
                    packed_bytes += weight.numel() * weight.element_size()
                    stack.enter_context(patch.object(mod, method, make(mod, weight)))

            # Also check the changed operation itself using the real middle layer input.
            middle = layers[len(layers) // 2]
            target = middle.nar_self_attn if name == "qkv" else middle.nar_mlp
            attr = "project_qkv" if name == "qkv" else "forward"
            original = getattr(target, attr)
            args_seen = []

            def capture(*args):
                args_seen.append(args)
                return original(*args)

            with patch.object(target, attr, capture):
                engine.velocity(state, 0.0)
            args = args_seen[0]
            local_ref = original(*args)
            _, packing = measure(install)
            packed_alloc = torch.mps.current_allocated_memory()
            local = getattr(target, attr)(*args)
            if name == "qkv":
                local_diff = [difference(x, y) for x, y in zip(local, local_ref)]
            else:
                local_diff = [difference(local, local_ref)]
            _, cold = measure(lambda: engine.velocity(state, 0.0))
            result, elapsed = measure(lambda: engine.velocity(state, 0.0))
            report["candidates"][name] = dict(
                pack_s=packing,
                packed_bytes=packed_bytes,
                allocated_delta=packed_alloc - before,
                cold_s=cold,
                velocity_s=elapsed,
                local_difference=local_diff,
                difference=difference(result, reference),
            )
            write_json(output / "result.json", report)
        torch.mps.empty_cache()
    _, report["baseline_end_s"] = measure(lambda: engine.velocity(state, 0.0))
    write_json(output / "result.json", report)
    engine.close()


def seed_methods(names):
    """Load just the reviewed patched methods from a temporary pinned source tree."""
    import ast
    import subprocess
    import tempfile
    from modules import diffusion_transformer, flow_matching

    upstream = ROOT / "third_party/seed-vc"
    revision = "51383efd921027683c89e5348211d93ff12ac2a8"
    if (
        subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        != revision
    ):
        raise ValueError("Seed-VC revision mismatch")
    specs = {
        "key-mask": (diffusion_transformer, "DiT", "forward"),
        "equal-heads": (diffusion_transformer, "Attention", "forward"),
        "loop": (flow_matching, "BASECFM", "solve_euler"),
    }
    stack = ExitStack()
    try:
        for name in names:
            module, cls, method = specs[name]
            rel = Path(module.__file__).relative_to(upstream)
            with tempfile.TemporaryDirectory() as directory:
                p = Path(directory) / rel
                p.parent.mkdir(parents=True)
                p.write_bytes(Path(module.__file__).read_bytes())
                subprocess.run(
                    ["git", "apply", str(ROOT / f"patches/candidates/seed-vc-{name}.patch")],
                    cwd=directory,
                    check=True,
                )
                node = next(
                    n
                    for n in ast.parse(p.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == cls
                )
                fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
                variables = dict(vars(module))
                exec(compile(ast.Module([fn], []), str(rel), "exec"), variables)
            stack.enter_context(patch.object(getattr(module, cls), method, variables[method]))
        return stack
    except BaseException:
        stack.close()
        raise


def mask_reference(device):
    generator = torch.Generator().manual_seed(73)
    q, k, v = [torch.randn(2, 4, 64, 64, generator=generator) for _ in range(3)]
    for lengths in [(64, 64), (64, 37)]:
        keymask = torch.arange(64)[None, :] < torch.tensor(lengths)[:, None]
        compact = keymask[:, None, None, :]
        dense = compact.repeat(1, 1, 64, 1)
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=dense)
        dtype = torch.float16 if device == "mps" else torch.float32
        inputs = [t.to(device, dtype) for t in (q, k, v)]
        a, b = [
            F.scaled_dot_product_attention(*inputs, attn_mask=mask.to(device)).float().cpu()
            for mask in (compact, dense)
        ]
        torch.testing.assert_close(a, b, atol=0.002, rtol=0.005)
        torch.testing.assert_close(a, ref, atol=0.002, rtol=0.005)


def seed_cost(run, output):
    from profile_seed_phrase import load_pair

    mask_reference("mps")
    start = time.perf_counter()
    cfm, vocoder = load_pair()
    del vocoder
    dit = cfm.to("mps").estimator
    dit.setup_caches(max_batch_size=1, max_seq_length=8192)
    torch.mps.synchronize()
    load_s = time.perf_counter() - start
    saved = torch.load(output.parent / "08_seed/dit-inputs.pt", map_location="cpu", weights_only=True)
    x, prompt, length, t, style, mu = (tensor.to("mps") for tensor in saved["args"])
    originals = [p[:1] for p in (prompt, style, mu)]

    def prepare():
        return [torch.cat([p, torch.zeros_like(p)]) for p in originals]

    fixed = prepare()
    for a, b in zip(fixed, (prompt, style, mu)):
        assert torch.equal(a, b)

    def call(reuse):
        prompt_cfg, style_cfg, mu_cfg = fixed if reuse else prepare()
        return dit(
            torch.cat((x[:1], x[:1])), prompt_cfg, length, torch.cat((t[:1], t[:1])), style_cfg, mu_cfg
        )

    report = {
        "shared_load_seconds": load_s,
        "variants": {},
        "mask_reference": "full and padded keys passed",
    }
    with torch.autocast("mps", dtype=torch.float16):
        reference, _ = measure(lambda: call(False))
        for name in ("baseline", "key-mask", "equal-heads", "loop", "baseline-end"):
            with seed_methods([name] if name in ("key-mask", "equal-heads") else []):
                actual, _ = measure(lambda: call(name == "loop"))
                _, t1 = measure(lambda: call(name == "loop"))
                _, t2 = measure(lambda: call(name == "loop"))
                report["variants"][name] = dict(seconds=[t1, t2], diff=difference(actual, reference))
            write_json(output / "result.json", report)
    # Credit a candidate only if both timings beat every bracketing baseline;
    # allow BF16/FP16 rounding at the previously measured FP16 quality scale.
    base = min(report["variants"][b]["seconds"][i] for b in ("baseline", "baseline-end") for i in (0, 1))
    report["winners"] = [
        n
        for n in ("key-mask", "equal-heads", "loop")
        if max(report["variants"][n]["seconds"]) < base
        and report["variants"][n]["diff"]["relative_rms"] < 0.000526
    ]
    write_json(output / "result.json", report)


def ar_combo(run, output):
    """Native GQA baseline versus the same model with a semantic-only head."""
    from yue2.modeling_yue2 import StaticKVCache
    from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END, Sampling
    from yue2.sampling import distribution
    from contextlib import nullcontext

    (model, prefix, codec), load_s = measure(lambda: yue_inputs(run))
    cfg = model.config
    assert MUSIC_END + 1 == CODEC_OFFSET
    weight = model.lm_head.weight
    allowed = slice(MUSIC_END, CODEC_OFFSET + CODEC_SIZE)
    folder = artifact(run, "02_cover", "cover").parent
    sampling = Sampling(**json.loads((folder / "config.json").read_text())["generation"]["semantic"])

    def native(q, k, v, *, attn_mask=None, is_causal=False):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=q.shape[1] != k.shape[1]
        )

    def compact(x):
        partial = F.linear(x, weight[allowed])
        logits = partial.new_zeros((*partial.shape[:-1], weight.shape[0]))
        logits[..., allowed] = partial
        return logits

    codec = [int(t) + CODEC_OFFSET for t in codec]
    report = {}
    with patch("yue2.modeling_yue2.sdpa", native):
        for count in (0, len(codec) - 16):
            ids = prefix + codec[:count]
            cache = StaticKVCache(
                cfg.num_hidden_layers,
                1,
                cfg.num_key_value_heads,
                len(ids) + 16,
                cfg.head_dim,
                torch.bfloat16,
                torch.device("mps"),
            )
            torch.mps.synchronize()
            prefill_start = time.perf_counter()
            for start in range(0, len(ids), 1024):
                model(
                    torch.tensor([ids[start : start + 1024]], device="mps"),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
            torch.mps.synchronize()
            prefill_s = time.perf_counter() - prefill_start
            tokens = [torch.tensor([[t]], device="mps") for t in codec[count : count + 8]]
            pos = cache._seen_tokens
            histories = [
                codec[max(0, count + i + 1 - sampling.penalty_window) : count + i + 1] for i in range(8)
            ]

            def replay(check=False):
                cache._seen_tokens = pos
                rng = torch.Generator().manual_seed(42)
                snapshots = []
                for i, token in enumerate(tokens):
                    logits = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[
                        :, -1
                    ]
                    probs = (
                        distribution(logits, sampling, histories[i], count + i + 1, "semantic")
                        .softmax(-1)
                        .cpu()
                    )
                    chosen = torch.multinomial(probs, 1, generator=rng)
                    token_id = chosen.item()
                    chosen.to("mps")
                    if check:
                        snapshots.append((logits[:, allowed].cpu(), probs, token_id))
                return snapshots, rng.get_state()

            timings = {"full": [], "compact": []}
            refs, state0 = replay(True)
            with patch.object(model.lm_head, "forward", compact):
                test, state1 = replay(True)
            checks = dict(
                finite=all(torch.isfinite(b[0]).all().item() for b in test),
                allowed_logits_equal=all(torch.equal(a[0], b[0]) for a, b in zip(refs, test)),
                probabilities_equal=all(torch.equal(a[1], b[1]) for a, b in zip(refs, test)),
                greedy_equal=all(a[1].argmax().item() == b[1].argmax().item() for a, b in zip(refs, test)),
                sampled_tokens_equal=all(a[2] == b[2] for a, b in zip(refs, test)),
                rng_equal=torch.equal(state0, state1),
            )
            for name in ("full", "compact", "compact", "full"):
                with (
                    patch.object(model.lm_head, "forward", compact) if name == "compact" else nullcontext()
                ):
                    _, elapsed = measure(replay)
                timings[name].append(elapsed / len(tokens))
            report[str(pos)] = dict(
                exact=all(checks.values()),
                checks=checks,
                seconds_per_token=timings,
                steps=len(tokens),
                shared_load_seconds=load_s,
                prefill_seconds=prefill_s,
            )
            write_json(output / "result.json", report)
            del cache
            torch.mps.empty_cache()
    if not all(c["exact"] for c in report.values()):
        raise FloatingPointError("Real decode candidate differs")


def seed_combo(run, output):
    from contextlib import nullcontext
    from profile_seed_phrase import run_phrase

    ledger = json.loads((output.parent / "budget.json").read_text())
    entries = [w for w in ledger["workers"] if w["mode"] == "seed-cost" and w["status"] == "passed"]
    if not entries:
        raise ValueError("No completed long-block screening")
    cmd = entries[-1]["command"]
    screen = json.loads((Path(cmd[cmd.index("--output") + 1]) / "result.json").read_text())
    winners = screen["winners"]
    if not winners:
        raise ValueError("No candidates passed the long-block screening")
    write_json(output / "selection.json", {"winners": winners, "source": cmd[cmd.index("--output") + 1]})
    run_phrase(
        run,
        output,
        variants=[
            ("fp16-baseline", True, nullcontext),
            ("fp16-candidate", True, lambda: seed_methods(winners)),
        ],
    )
