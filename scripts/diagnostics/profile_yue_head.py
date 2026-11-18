"""Paired MPS timing of the two semantic output projections and full sampler."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]


def run_head(run, output):
    from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END, Sampling
    from yue2.sampling import distribution
    cover = Path(json.loads((run/'stage_state.json').read_text())['stages']['02_cover']['outputs']['cover']['path']).parent
    sampling = Sampling(**json.loads((cover/'config.json').read_text())['generation']['semantic'])
    ids = np.load(cover/'semantic.npy')
    history = [int(t)+CODEC_OFFSET for t in ids[:500] if 0 <= t < CODEC_SIZE]
    with safe_open(ROOT/'models/YuE2-3B/model.safetensors',framework='pt',device='cpu') as f:
        w = f.get_tensor('lm_head.weight').to('mps')
    assert MUSIC_END+1 == CODEC_OFFSET and w.dtype == torch.bfloat16
    allowed = slice(MUSIC_END, CODEC_OFFSET+CODEC_SIZE)
    sub = w[allowed]
    h = torch.randn(1,w.shape[1],generator=torch.Generator().manual_seed(937)).to(w)
    def linear(hidden, compact):
        return torch.nn.functional.linear(hidden, sub if compact else w)
    def project(hidden, compact):
        z = linear(hidden,compact)
        if not compact:
            return z
        scores = z.new_zeros((1,w.shape[0]))
        scores[:,allowed] = z
        return scores
    def probabilities(hidden, compact, step):
        return distribution(project(hidden,compact), sampling, history, step,'semantic').softmax(-1).cpu()
    def sample(hidden, compact, generator):
        prob = probabilities(hidden, compact, len(history))
        idx = torch.multinomial(prob,1,generator=generator)
        token = int(idx.item())
        return token, idx.to('mps')
    def elapsed(fn, count):
        torch.mps.synchronize()
        begin = time.perf_counter()
        for _ in range(count):
            fn()
        torch.mps.synchronize()
        return (time.perf_counter()-begin)/count
    checks = []
    for vec, step in [(h,len(history)), (w[MUSIC_END:MUSIC_END+1], 0),
                       (w[MUSIC_END:MUSIC_END+1], sampling.min_tokens)]:
        a,b = (project(vec,k)[:,allowed].float().cpu() for k in (False,True))
        p,q = (probabilities(vec,k,step) for k in (False,True))
        ga,gb = [torch.Generator().manual_seed(42) for _ in range(2)]
        choices = [torch.multinomial(v,1,generator=g).item() for v,g in ((p,ga),(q,gb))]
        checks.append(dict(logit_error=float((a-b).abs().max()),prob_error=float((p-q).abs().max()),
                           finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(q).all()),
                           greedy=[int(p.argmax()),int(q.argmax())],eos_p=[float(p[0,MUSIC_END]),float(q[0,MUSIC_END])],
                           choices=choices,rng_equal=torch.equal(ga.get_state(),gb.get_state())))
    result = dict(checks=checks, batches=[], count_per_batch=8)
    def save():
        (output/'head.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    for compact in (False,True):
        sample(h,compact,torch.Generator().manual_seed(42))
    for compact in (False,True,True,False):
        g = torch.Generator().manual_seed(42)
        raw = elapsed(lambda: linear(h,compact),8)
        total = elapsed(lambda: sample(h,compact,g),8)
        result['batches'].append(dict(compact=compact,projection_s=raw,net_s=total))
        save()
    passed = all(r['finite'] and r['logit_error']==r['prob_error']==0 and r['choices'][0]==r['choices'][1]
                 and r['greedy'][0]==r['greedy'][1] and r['rng_equal'] for r in checks)
    passed = passed and checks[1]['eos_p']==[0.,0.] and checks[2]['greedy']==[MUSIC_END]*2
    result['numeric_passed'] = passed
    save()
    if not passed:
        raise ValueError('Semantic head differs; retain original projection')
