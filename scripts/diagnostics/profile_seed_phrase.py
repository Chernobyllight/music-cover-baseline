"""Prepare a fixed vocal phrase on CPU and replay the original 30-step solver."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
RATE, HOP, FRAMES = 44100, 512, 176


def load_pair():
    import yaml
    sys.path.insert(0, str(ROOT / 'third_party/seed-vc'))
    from modules.commons import recursive_munch
    from modules.flow_matching import CFM
    from modules.bigvgan.bigvgan import BigVGAN
    params = recursive_munch(yaml.safe_load((ROOT / 'configs/seed_svc.local.yml').read_text())['model_params'])
    cfm = CFM(params)
    weights = torch.load(next((ROOT/'models/Seed-VC').glob('*.pth')), map_location='cpu',
                         weights_only=True, mmap=True)['net']['cfm']
    weights = {k.removeprefix('module.'): v for k, v in weights.items()}
    # The pinned loader filters other head architectures and regenerates position
    # buffers. Require every learned parameter used by this configuration.
    target = cfm.state_dict()
    matched = {k: v for k, v in weights.items() if k in target and v.shape == target[k].shape}
    missing = set(dict(cfm.named_parameters())) - matched.keys()
    if missing:
        raise ValueError(f"Missing learned DiT weights: {sorted(missing)}")
    cfm.load_state_dict(matched, strict=False)
    vocoder = BigVGAN.from_pretrained(str(ROOT/'models/BigVGAN'), use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    return cfm.eval(), vocoder.eval()


def pitch(audio):
    import librosa
    signal = librosa.resample(audio, orig_sr=RATE, target_sr=16000)
    f0, voiced, _ = librosa.pyin(signal, sr=16000, fmin=65, fmax=1000,
                                 frame_length=1024, hop_length=160)
    return np.where(voiced, f0, np.nan)


def prepare(ledger):
    saved = torch.load(ledger/'08_seed/dit-inputs.pt', map_location='cpu', weights_only=True)
    x, prompt, lengths, t, style, mu = saved['args']
    ref = int(prompt[0].ne(0).any(0).sum())
    assert torch.count_nonzero(t) == 0 and ref == 898 and int(lengths[0]) == x.shape[-1]
    assert not prompt[0,:,ref:].any() and not x[0,:,:ref].any() and torch.equal(x[0], x[1])
    audio, sr = sf.read(ledger/'08_seed/source-20s.wav', dtype='float32')
    assert sr == RATE and audio.ndim == 1
    f0 = pitch(audio)
    def score(start):
        a, b = (round(i*HOP/RATE*100) for i in (start, start+FRAMES))
        voiced = np.isfinite(f0[a:b]).mean()
        rms = np.sqrt(np.mean(audio[start*HOP:(start+FRAMES)*HOP]**2))
        return voiced * rms
    start = max(range(x.shape[-1]-ref-FRAMES+1), key=score)
    if score(start) <= 0:
        raise ValueError('No voiced phrase in the saved source')
    idx = torch.cat((torch.arange(ref), torch.arange(ref+start,ref+start+FRAMES)))
    data = dict(x=x[:1,:,idx], prompt=prompt[:1,:,:ref], mu=mu[:1,idx], style=style[:1],
                start=start, ref=ref)
    assert all(torch.isfinite(data[k]).all() for k in ('x','prompt','mu','style'))
    load_pair()  # CPU checkpoint and architecture compatibility
    torch.save(data, ledger/'seed-phrase.pt')
    sf.write(ledger/'seed-phrase-source.wav', audio[start*HOP:(start+FRAMES)*HOP], RATE, subtype='FLOAT')
    print('source start/end seconds',start*HOP/RATE,(start+FRAMES)*HOP/RATE,'reference frames',ref)


def run_phrase(run, output, *, variants=None):
    from contextlib import nullcontext
    from unittest.mock import patch
    data = torch.load(output.parent/'seed-phrase.pt', weights_only=True, map_location='cpu')
    begin = time.perf_counter()
    cfm, vocoder = (m.to('mps') for m in load_pair())
    cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    x, prompt, mu, style = (data[k].to('mps') for k in ('x','prompt','mu','style'))
    grid = torch.linspace(0, 1, 31, device='mps')
    lengths = torch.tensor([x.shape[-1]], device='mps')
    ref = data['ref']
    torch.mps.synchronize()
    report = dict(load_seconds=time.perf_counter()-begin, ref_frames=ref,
                  source_start_frame=data['start'], source_frames=x.shape[-1]-ref, variants={})
    def save():
        (output/'phrase.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    paired_fp16 = variants is not None
    if variants is None:
        variants = [('fp32', False, nullcontext), ('fp16', True, nullcontext)]
    for name, half, context in variants:
        with context():
            estimator = cfm.estimator.forward
            states = []
            def forward(*args, **kwargs):
                states.append(args[0][:1,:,ref:])  # CFG stack is newly allocated each Euler step
                with torch.autocast('mps', dtype=torch.float16, enabled=half):
                    return estimator(*args, **kwargs)
            with patch.object(cfm.estimator, 'forward', forward):
                # Keep autocast's weight cache alive across all calls; only DiT runs in half.
                with torch.autocast('mps', dtype=torch.float16, enabled=half), torch.autocast('mps', enabled=False):
                    warm_s = None
                    if paired_fp16:
                        # Warm each implementation at the same phrase shape, including
                        # its CFG construction and the common FP32 vocoder.
                        torch.mps.synchronize()
                        warm_start = time.perf_counter()
                        probe = cfm.solve_euler(x.clone(), lengths, prompt, mu.clone(), style, None, grid[:2], 0.7)
                        vocoder(probe[:, :, ref:].float())
                        torch.mps.synchronize()
                        warm_s = time.perf_counter() - warm_start
                        states.clear()
                        del probe
                    initial = x.clone()
                    torch.mps.synchronize()
                    begin = time.perf_counter()
                    mel = cfm.solve_euler(initial, lengths, prompt, mu.clone(), style, None, grid, 0.7)[:,:,ref:]
                    torch.mps.synchronize()
            solve_s = time.perf_counter()-begin
            assert len(states) == 30 and mel.dtype == torch.float32
            trace = torch.stack(states[1:]+[mel]).float().cpu().numpy()
            assert np.isfinite(trace).all()
            np.save(output/f'{name}-trajectory.npy',trace)
            begin = time.perf_counter()
            wave = vocoder(mel.float()).reshape(-1)
            torch.mps.synchronize()
            decode_s = time.perf_counter()-begin
            wave = wave.cpu().numpy()
            assert len(wave) == FRAMES*HOP and np.isfinite(wave).all()
            sf.write(output/f'{name}.wav',wave,RATE,subtype='FLOAT')
            report['variants'][name] = dict(solve_s=solve_s, decode_s=decode_s, calls=len(states),
                                           samples=len(wave), peak=float(np.max(np.abs(wave))))
            if warm_s is not None:
                report['variants'][name]['warm_s'] = warm_s
            save()


def compare(output, names=('fp32','fp16'), accepted=None):
    tracks = {}
    for name in names:
        wave, rate = sf.read(output/f'{name}.wav', dtype='float32')
        if rate != RATE or wave.shape != (FRAMES*HOP,) or not np.isfinite(wave).all():
            raise ValueError(f'Invalid output: {name}')
        tracks[name] = pitch(wave)
    a,b = tracks.values()
    voiced = np.isfinite(a) & np.isfinite(b)
    if not np.any(voiced):
        raise ValueError('No jointly voiced F0 samples')
    cents = np.abs(1200*np.log2(a[voiced]/b[voiced]))
    tensors = [np.load(output/f'{name}-trajectory.npy').astype(np.float64) for name in tracks]
    a_mel,b_mel = (x.reshape(30,-1) for x in tensors)
    delta = a_mel-b_mel
    report = dict(frames=len(a), voiced_both=int(voiced.sum()),
                  voiced_disagreement=int((np.isfinite(a)!=np.isfinite(b)).sum()),
                  f0_abs_cents_median=float(np.median(cents)), f0_abs_cents_p95=float(np.percentile(cents,95)),
                  f0_abs_cents_max=float(cents.max()), f0_method='pYIN 16kHz, 10ms hop, 10-cent bins',
                  step_relative_rms=(np.linalg.norm(delta,axis=1)/np.linalg.norm(a_mel,axis=1)).tolist(),
                  step_max_abs=np.abs(delta).max(axis=1).tolist())
    if accepted is not None:
        import hashlib
        reference, reference_rate = sf.read(accepted, dtype='float32')
        if reference_rate != RATE or reference.shape != (FRAMES*HOP,):
            raise ValueError(f'Invalid accepted reference: {accepted}')
        matches = {}
        for path in [accepted, *(output/f'{name}.wav' for name in names)]:
            audio, _ = sf.read(path, dtype='float32')
            matches[path.stem] = dict(sha256_float32_le=hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest(),
                                      equal=bool(np.array_equal(reference,audio)),
                                      max_abs=float(np.abs(reference-audio).max()))
        report['decoded_pcm_vs_previously_accepted'] = matches
    (output/'quality.json').write_text(json.dumps(report,indent=2)+chr(10))
    print(json.dumps({k:v for k,v in report.items() if not k.startswith('step_')},indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','compare'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--names', nargs=2, default=('fp32', 'fp16'), help='WAV stems for compare')
    parser.add_argument('--accepted', type=Path, help='Optional prior reference WAV for PCM comparison')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with torch.inference_mode():
        if args.action == 'prepare':
            prepare(args.directory)
        else:
            compare(args.directory, names=args.names, accepted=args.accepted)
