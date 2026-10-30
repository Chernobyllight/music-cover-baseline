"""SVC 4.1 adapter: isolated imports, strict loading, explicit F0 and sample-aligned chunks."""
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from ..common.paths import ROOT
from ..common.io import write_json
from ..audio import sha256, inspect_audio
from ..svc import check_assets, check_model_config, checked_state, validate_settings, UPSTREAM_COMMIT


@contextlib.contextmanager
def runtime(workspace):
    workspace = Path(workspace).resolve(); workspace.mkdir(parents=True, exist_ok=True)
    pretrain = workspace/'pretrain'; pretrain.mkdir(exist_ok=True)
    for name, target in {'checkpoint_best_legacy_500.pt': ROOT/'models/SoVitsSVC/contentvec.pt',
                         'rmvpe.pt': ROOT/'models/SoVitsSVC/rmvpe.pt'}.items():
        p = pretrain/name
        if p.is_symlink():
            if p.resolve() != target.resolve(): raise ValueError('SVC runtime asset link changed')
        elif p.exists(): raise ValueError('Unexpected SVC runtime asset: '+str(p))
        else: p.symlink_to(target.resolve())
    previous = Path.cwd(); upstream = str(ROOT/'third_party/so-vits-svc')
    sys.path.insert(0, upstream); os.chdir(workspace)
    try: yield
    finally:
        os.chdir(previous); sys.path.remove(upstream)
        for name in ('checkpoint_best_legacy_500.pt', 'rmvpe.pt'):
            link = pretrain/name
            if link.is_symlink(): link.unlink()
        try: pretrain.rmdir(); workspace.rmdir()
        except OSError: pass


def build_model(config, checkpoint, device='cuda', replace_speaker=False):
    import torch
    from models import SynthesizerTrn
    check_model_config(config)
    model = SynthesizerTrn(config['data']['filter_length']//2+1,
                          config['train']['segment_size']//config['data']['hop_length'], **config['model'])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    coverage = checked_state(model, payload['model'], replace_speaker=replace_speaker)
    return model.to(device), coverage


class Converter:
    def __init__(self, checkpoint, config, speaker):
        import torch
        import utils
        check_model_config(config, speaker)
        self.torch = torch; self.utils = utils; self.config = config
        self.model, self.coverage = build_model(config, checkpoint)
        self.model.eval().requires_grad_(False)
        from .svc_contentvec import ContentVec
        self.encoder = ContentVec(device='cuda', variant=config['model']['speech_encoder'])
        self.f0 = utils.get_f0_predictor('rmvpe', hop_length=512, sampling_rate=44100, device='cuda', threshold=.05)
        self.sid = torch.tensor([[config['spk'][speaker]]], device='cuda', dtype=torch.long)

    def infer(self, source, output, settings=None, seed=42):
        import numpy as np
        import soundfile as sf
        import librosa
        import torch
        p = validate_settings(settings or {})
        wave, sr = sf.read(source, dtype='float32', always_2d=True)
        if sr != 44100: raise ValueError('SVC input must be decoded to 44.1 kHz')
        wave = wave.mean(axis=1)
        if not np.isfinite(wave).all() or not len(wave): raise ValueError('Invalid SVC input audio')
        width = round(p['chunk_seconds']*sr); overlap = round(p['overlap_seconds']*sr); pad = round(p['pad_seconds']*sr)
        result = np.zeros(len(wave), np.float32); weights = np.zeros(len(wave), np.float32)
        segments = []; start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for index, first in enumerate(range(0, len(wave), width-overlap)):
                last = min(len(wave), first+width); core = wave[first:last]
                if np.sqrt(np.mean(core**2)) < 1e-5:
                    audio = np.zeros(len(core), np.float32); status = 'silent'
                else:
                    # Real context on both sides; zeros only beyond original timeline.
                    block = np.pad(wave[max(0, first-pad):min(len(wave), last+pad)],
                                   (max(0, pad-first), max(0, last+pad-len(wave))))
                    f0, uv = self.f0.compute_f0_uv(block)
                    c = self.encoder.encoder(torch.from_numpy(librosa.resample(block, orig_sr=sr, target_sr=16000)).to('cuda'))
                    c = self.utils.repeat_expand_2d(c.squeeze(0), len(f0), mode=self.config['data']['unit_interpolate_mode'])[None]
                    t = lambda a: torch.tensor(a, device='cuda', dtype=torch.float32)[None]
                    generated, _ = self.model.infer(c, f0=t(f0), uv=t(uv), g=self.sid,
                        noice_scale=p['noise_scale'], seed=(seed+index) % 2**32, predict_f0=False)
                    audio = generated[0,0].float().cpu().numpy()
                    if abs(len(audio)-len(block)) > 1024: raise RuntimeError('SVC output frame count mismatch')
                    audio = np.pad(audio, (0,max(0,len(block)-len(audio))))[pad:pad+len(core)]
                    status = 'converted'
                weight = np.ones(len(core), np.float32)
                n = min(overlap, len(core))
                if first: weight[:n] *= np.linspace(1e-6,1,n)
                if last < len(wave): weight[-n:] *= np.linspace(1,1e-6,n)
                result[first:last] += audio * weight; weights[first:last] += weight
                segments.append({'first_sample':first,'last_sample':last,'status':status,'seed':(seed+index)%2**32})
                if last == len(wave): break
        result /= np.maximum(weights, 1e-12)
        if not np.isfinite(result).all(): raise RuntimeError('Nonfinite SVC output')
        peak = float(np.max(np.abs(result))); gain = min(1.0,.99/max(peak,1e-12))
        sf.write(output, result*gain, sr, subtype='FLOAT')
        return {'segments':segments,'settings':p,'pre_limit_peak':peak,'peak_gain':gain,
                'duration_s':len(wave)/sr,'wall_seconds':time.monotonic()-start,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(), 'parameter_loading':self.coverage}


def convert(request, output):
    from ..voice import resolve_voice, voice_record
    check_assets()
    voice = resolve_voice('finetuned', request['voice_profile'], backend='so_vits_svc')
    output = Path(output).resolve()
    with runtime(output/'runtime'):
        converter = Converter(voice['checkpoint'], json.loads(voice['config'].read_text()), voice['speaker_name'])
        report = converter.infer(request['source'], output/'converted.wav', request.get('svc'), request['seed'])
    write_json(output/'conversion.json', {**voice_record(voice), **report,
        'upstream_commit':UPSTREAM_COMMIT,'source_sha256':sha256(request['source']),
        'reference_sha256':sha256(request['reference']), 'seed':request['seed'],
        'semitones':0,'auto_f0_adjust':False,'training_performed':False,
        'reference_used_for_inference':False,'reference_role':'evaluation_and_pitch_adaptation',
        'diffusion_steps':None,'audio':inspect_audio(output/'converted.wav')})
