"""Pinned Fish Diffusion acoustic model and sample-aligned singing conversion."""
import json
import sys
import time
from pathlib import Path
from ..audio import sha256, inspect_audio
from ..common.paths import ROOT
from ..common.io import write_json
from ..fish import MODEL_SPEC, UPSTREAM_COMMIT, validate_settings, checked_state, check_assets, check_model_config


def runtime():
    """Only called in the dedicated base-Python worker, never by other backends."""
    import os
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('WANDB_MODE', 'disabled')
    import torch
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    upstream = str(ROOT/'third_party/fish-diffusion')
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    return torch


def acoustic_config():
    runtime()
    from mmengine import Config
    from fish_diffusion.utils.pitch import pitch_to_log
    config = Config.fromfile(str(ROOT/'third_party/fish-diffusion/configs/_base_/archs/diff_svc_v2.py'))
    config.model.speaker_encoder.input_size = 1
    config.model.pitch_encoder.preprocessing = pitch_to_log
    config.model.pitch_shift_encoder = dict(type='NaiveProjectionEncoder', input_size=1,
        output_size=256, use_embedding=False)
    return config


def build_model(checkpoint=None, *, pretrained=False, device='cuda'):
    torch = runtime()
    from fish_diffusion.archs.diffsinger.diffsinger import DiffSinger
    model = DiffSinger(acoustic_config().model)
    coverage = None
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        saved = payload.get('state_dict', payload.get('model', payload))
        # Released checkpoints may contain EMA and frozen vocoder tensors.
        if any(k.startswith('ema_model.') for k in saved):
            saved = {k[len('ema_model.'):]: v for k, v in saved.items() if k.startswith('ema_model.')}
        elif any(k.startswith('model.') for k in saved):
            saved = {k[len('model.'):]: v for k, v in saved.items() if k.startswith('model.')}
        coverage = checked_state(model, saved, replace_speaker=pretrained)
    return model.to(device), coverage


class FeatureExtractor:
    def __init__(self, device='cuda'):
        self.torch = runtime()
        from fish_diffusion.modules.feature_extractors.content_vec import ContentVec
        from fish_diffusion.modules.pitch_extractors.parsel_mouth import ParselMouthPitchExtractor
        from fish_diffusion.modules.vocoders.nsf_hifigan.nsf_hifigan import NsfHifiGAN
        self.device = device
        self.encoder = ContentVec(str(ROOT/'models/FishDiffusion/contentvec'), output_layer=9,
                                  use_projection=True).to(device).eval().requires_grad_(False)
        self.pitch = ParselMouthPitchExtractor(f0_min=40, f0_max=2000, keep_zeros=False)
        self.vocoder = NsfHifiGAN(str(ROOT/'models/FishDiffusion/nsf_hifigan/model'),
            use_natural_log=False, sampling_rate=44100, mel_channels=128).to(device).eval().requires_grad_(False)

    def extract(self, audio, key_shift=0.0):
        from fish_diffusion.utils.tensor import repeat_expand
        torch = self.torch
        wave = torch.as_tensor(audio, device=self.device, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            mel = self.vocoder.wav2spec(wave, 44100, key_shift=key_shift)
            length = mel.shape[-1]
            contents = repeat_expand(self.encoder(wave, 44100)[0], length)
            pitches = self.pitch(wave, 44100, pad_to=length) * (2**(key_shift/12))
        return {'mel': mel, 'contents': contents, 'pitches': pitches, 'key_shift': float(key_shift)}


class Converter:
    def __init__(self, checkpoint, config):
        check_model_config(config)
        self.torch = runtime()
        self.model, self.coverage = build_model(checkpoint)
        self.model.eval().requires_grad_(False)
        self.features = FeatureExtractor()

    def infer(self, source, output, settings=None, seed=42):
        import numpy as np
        import soundfile as sf
        from fish_diffusion.utils.tensor import repeat_expand
        p = validate_settings(settings or {})
        wave, sr = sf.read(source, dtype='float32', always_2d=True)
        if sr != 44100 or not len(wave) or not np.isfinite(wave).all():
            raise ValueError('Fish input must be finite 44.1 kHz audio')
        wave = wave.mean(axis=1)
        torch = self.torch
        width, overlap, pad = [round(p[k]*sr) for k in ('chunk_seconds', 'overlap_seconds', 'pad_seconds')]
        result, weights = np.zeros_like(wave), np.zeros_like(wave)
        segments = []
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for index, first in enumerate(range(0, len(wave), width-overlap)):
                last = min(len(wave), first+width)
                core = wave[first:last]
                chunk_seed = (seed+index) % 2**32
                torch.manual_seed(chunk_seed)
                if np.sqrt(np.mean(core**2)) < 1e-5:
                    audio, status = np.zeros_like(core), 'silent'
                else:
                    block = np.pad(wave[max(0,first-pad):min(len(wave),last+pad)],
                                   (max(0,pad-first), max(0,last+pad-len(wave))))
                    # Match upstream inference frame count; retain original timeline.
                    x = torch.from_numpy(block).to('cuda')[None]
                    length = len(block)//512
                    pitches = self.features.pitch(x, sr, pad_to=length)
                    if not torch.any(pitches > 0):
                        audio, status = np.zeros_like(core), 'no_voiced_frames'
                    else:
                        content = repeat_expand(self.features.encoder(x, sr)[0], length).T[None]
                        lengths = torch.tensor([length], device='cuda')
                        condition = self.model.forward_features(speakers=torch.tensor([0], device='cuda'),
                            contents=content, contents_lens=lengths, contents_max_len=length,
                            mel_lens=lengths, mel_max_len=length, pitches=pitches[None,:,None],
                            pitch_shift=torch.zeros((1,1), device='cuda'))
                        mel = self.model.diffusion(condition['features'], sampler_interval=p['sampler_interval'],
                            noise_predictor=p['noise_predictor'], progress=False)
                        generated = self.features.vocoder.spec2wav(mel[0].T, pitches).float().cpu().numpy()
                        if abs(len(generated)-len(block)) > 1024:
                            raise RuntimeError('Fish output frame count mismatch')
                        generated = np.pad(generated, (0,max(0,len(block)-len(generated))))
                        audio, status = generated[pad:pad+len(core)], 'converted'
                weight = np.ones(len(core), np.float32)
                n = min(overlap,len(core))
                if first: weight[:n] *= np.linspace(1e-6,1,n)
                if last < len(wave): weight[-n:] *= np.linspace(1,1e-6,n)
                result[first:last] += audio*weight
                weights[first:last] += weight
                segments.append({'first_sample':first,'last_sample':last,'seed':chunk_seed,'status':status})
                print(f'Fish chunk {index+1}: {first/sr:.2f}–{last/sr:.2f}s {status}', flush=True)
                if last == len(wave): break
        if np.any(weights <= 0): raise RuntimeError('Uncovered output timeline')
        result /= weights
        if not np.isfinite(result).all(): raise RuntimeError('Nonfinite Fish output')
        peak = float(np.max(np.abs(result)))
        gain = min(1.0, .99/max(peak,1e-12))
        sf.write(output, result*gain, sr, subtype='FLOAT')
        return {'settings':p, 'segments':segments, 'pre_limit_peak':peak, 'peak_gain':gain,
                'duration_s':len(wave)/sr, 'wall_seconds':time.monotonic()-start,
                'parameter_loading':self.coverage, 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}


def convert(request, output):
    from ..voice import resolve_voice, voice_record
    check_assets()
    voice = resolve_voice('finetuned', request['voice_profile'], backend='fish_diffusion')
    converter = Converter(voice['checkpoint'], json.loads(voice['config'].read_text()))
    report = converter.infer(request['source'], output/'converted.wav', request.get('fish'), request['seed'])
    write_json(output/'conversion.json', {**voice_record(voice), **report,
        'runtime':{'python':sys.executable,'prefix':sys.prefix,'torch':converter.torch.__version__},
        'source_sha256':sha256(request['source']), 'reference_sha256':sha256(request['reference']),
        'seed':request['seed'], 'semitones':0, 'auto_f0_adjust':False,
        'training_performed':False, 'reference_used_for_inference':False,
        'reference_role':'evaluation_only_at_stage_05', 'diffusion_timesteps':1000,
        'upstream_commit':UPSTREAM_COMMIT, 'audio':inspect_audio(output/'converted.wav')})
