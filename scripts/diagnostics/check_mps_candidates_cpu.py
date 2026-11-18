"""CPU-only numerical screens. Run with each candidate's own Python."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]


def yue():
    from safetensors import safe_open
    from yue2.protocol import ABC_END, CODEC_OFFSET, CODEC_SIZE, MUSIC_END, Sampling
    from yue2.sampling import distribution

    with safe_open(ROOT/'models/YuE2-3B/model.safetensors', framework='pt', device='cpu') as weights:
        weight = weights.get_tensor('lm_head.weight')
    assert MUSIC_END == CODEC_OFFSET - 1
    allowed = slice(MUSIC_END, CODEC_OFFSET + CODEC_SIZE)
    rng = torch.Generator().manual_seed(937)
    # One row per branch, matching unbatched positive/negative decode.
    hidden = [torch.randn(1, weight.shape[1], generator=rng).to(weight.dtype) for _ in range(2)]
    full = [torch.nn.functional.linear(x, weight) for x in hidden]
    local = [torch.nn.functional.linear(x, weight[allowed]) for x in hidden]
    report = {'dtype': str(weight.dtype), 'shape': list(weight.shape), 'cases': []}
    repeated = int(full[0][0, CODEC_OFFSET:allowed.stop].argmax()) + CODEC_OFFSET
    for scale in (1., 1.5):
        original, compact = (pair[0] if scale == 1 else pair[1]+scale*(pair[0]-pair[1]) for pair in (full, local))
        restored = torch.zeros_like(original)
        restored[:, allowed] = compact
        for step in (0, 250):
            for legacy in (False, True):
                args = (Sampling(), [repeated]*3, step, 'semantic', legacy)
                a, b = (distribution(x, *args).softmax(-1) for x in (original, restored))
                g1, g2 = [torch.Generator().manual_seed(42) for _ in range(2)]
                drawn = [torch.multinomial(p, 1, generator=g) for p,g in ((a,g1),(b,g2))]
                report['cases'].append(dict(cfg=scale,step=step,legacy=legacy,
                    allowed_logit_max_error=float((original[:,allowed]-compact).abs().max()),
                    probability_max_error=float((a-b).abs().max()), probabilities_finite=bool(torch.isfinite(b).all()),
                    token_equal=bool(torch.equal(*drawn)), rng_equal=torch.equal(g1.get_state(),g2.get_state()),
                    eos_probability=float(b[0,MUSIC_END])))
    # A hidden vector aligned with the actual EOS row exercises min_tokens
    # on a case where EOS would otherwise win greedy decoding.
    eos_vector = weight[MUSIC_END:MUSIC_END+1].clone()
    logits = torch.nn.functional.linear(eos_vector, weight)
    restored = torch.zeros_like(logits)
    restored[:, allowed] = torch.nn.functional.linear(eos_vector, weight[allowed])
    report['greedy_eos'] = {}
    for step in (0, 250):
        args = (Sampling(temperature=0), [], step, 'semantic')
        choices = [int(distribution(x, *args).argmax()) for x in (logits, restored)]
        assert choices[0] == choices[1] and (choices[0] == MUSIC_END) == (step >= 200)
        report['greedy_eos'][str(step)] = choices[0]
    config = json.loads((ROOT/'models/YuE2-3B/config.json').read_text())
    h = config['hidden_size']
    kv = config['num_key_value_heads'] * config['head_dim']
    layer_weights = 2*h*h + 2*h*kv + 3*h*config['intermediate_size']
    total = config['num_hidden_layers']*layer_weights + weight.numel()
    report['projection_fraction_of_dense_AR_macs'] = weight.numel()/total
    report['saved_fraction_of_dense_AR_macs'] = (weight.numel()-weight[allowed].numel())/total
    assert ABC_END < MUSIC_END  # ABC retains its full original projection.
    return report


def seed():
    revision = '51383efd921027683c89e5348211d93ff12ac2a8'
    upstream = ROOT/'third_party/seed-vc'
    source = subprocess.check_output(['git','-C',str(upstream),'show',revision+':modules/flow_matching.py'],text=True)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'modules/flow_matching.py'
        path.parent.mkdir()
        path.write_text(source)
        subprocess.run(['git','apply',str(ROOT/'patches/candidates/seed-vc-loop.patch')],cwd=temp,check=True)
        candidate = path.read_text()
    def solver(source):
        fn = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name=='solve_euler')
        scope = {'torch': torch, 'tqdm': lambda items: items}
        exec(compile(ast.Module([fn],[]),'solve_euler','exec'), scope)
        return scope['solve_euler']
    saved = torch.load(ROOT/'runs/probes/mps-performance/08_seed/dit-inputs.pt',weights_only=True,map_location='cpu')
    x, prompt, lengths, t, style, mu = saved['args']
    assert torch.equal(x[:1],x[1:]) and all(torch.count_nonzero(p[1:]) == 0 for p in (prompt, style, mu))
    state_bytes = x[:1].numel()*x.element_size()
    static_bytes = sum(p.numel()*p.element_size() for p in (prompt,style,mu))
    class Example:
        def estimator(self, x,prompt,length,t,style,mu):
            return x*.5 + prompt + t[:,None,None] + style[:,:1,None] + mu[:,:,:1].transpose(1,2)
    results = []
    for cfg in (0.,.7):
        for zero in (False,True):
            example = Example()
            example.zero_prompt_speech_token = zero
            inputs = (x[:1,:2,:12],torch.tensor([12]),torch.ones(1,2,4),mu[:1,:12,:2],style[:1,:2],None,torch.linspace(0,1,31))
            def call(src):
                return solver(src)(example, *(p.clone() if isinstance(p,torch.Tensor) else p for p in inputs),cfg)
            a,b = call(source),call(candidate)
            assert torch.equal(a,b) and not torch.count_nonzero(b[:,:,:4])
            results.append(dict(cfg=cfg,zero_prompt=zero,final_equal=True,prompt_zero=True))
    return dict(cases=results,static_cat_output_bytes_per_chunk_saved=29*static_bytes,
                static_zero_fill_bytes_per_chunk_saved=29*static_bytes//2,
                obsolete_states_bytes_at_final_step=29*state_bytes,shape=list(x.shape))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', choices=['seed','yue'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with torch.inference_mode():
        print(json.dumps((seed if args.candidate=='seed' else yue)(),indent=2))
