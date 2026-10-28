"""Independent speaker similarity plus aligned pitch diagnostics, not a listening verdict."""
import json
import sys
from pathlib import Path
import numpy as np
from ..audio import read_at, inspect_audio, sha256, estimate_alignment, shift_audio
from ..common.io import write_json
from ..common.paths import ROOT


class VoiceMetrics:
    def __init__(self):
        import torch
        from speechbrain.lobes.features import Fbank
        from speechbrain.processing.features import InputNormalization
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        self.torch = torch
        torch.set_num_threads(4)
        assets = json.loads((ROOT / 'configs/voice_training_assets.lock.json').read_text())['assets']
        for a in assets:
            if a['scope'] == 'evaluation' and (not (ROOT / a['path']).is_file() or sha256(ROOT / a['path']) != a['sha256']):
                raise ValueError('Independent voice evaluation assets missing/corrupt; run setup_voice_training.py --scope evaluation')
        self.features = Fbank(n_mels=80).eval().to('cuda')
        self.normalizer = InputNormalization(norm_type='sentence', std_norm=False).eval().to('cuda')
        self.encoder = ECAPA_TDNN(input_size=80, channels=[1024,1024,1024,1024,3072],
                                 kernel_sizes=[5,3,3,3,1], dilations=[1,2,3,4,1], attention_channels=128, lin_neurons=192)
        self.encoder.load_state_dict(torch.load(ROOT / 'models/voice_evaluation/ecapa/embedding_model.ckpt', map_location='cpu', weights_only=False))
        self.encoder.eval().requires_grad_(False).to('cuda')
        sys.path.insert(0, str(ROOT / 'third_party/seed-vc'))
        from modules.rmvpe import RMVPE
        self.rmvpe = RMVPE(str(ROOT / 'models/RMVPE/rmvpe.pt'), is_half=False, device='cuda')
        self.embeddings = {}

    def embedding(self, path):
        key = sha256(path)
        if key in self.embeddings: return self.embeddings[key]
        y = read_at(path, 16000).mean(axis=1)
        vectors = []
        with self.torch.no_grad():
            for start in range(0, len(y), 16000 * 12):
                part = y[start:start + 16000 * 12]
                if len(part) < 16000 or np.sqrt(np.mean(part ** 2)) < 1e-4: continue
                wave = self.torch.from_numpy(part).unsqueeze(0).to('cuda')
                lens = self.torch.ones(1, device='cuda')
                feat = self.normalizer(self.features(wave), lens)
                v = self.encoder(feat, lens).flatten().cpu().numpy()
                vectors.append(v / max(np.linalg.norm(v), 1e-12))
        if not vectors: raise ValueError(f'Insufficient active audio for speaker evaluation: {path}')
        value = np.mean(vectors, axis=0)
        value /= max(np.linalg.norm(value), 1e-12)
        self.embeddings[key] = value
        return value

    def pitch(self, y, target):
        import librosa
        from ..pitch_selection import load_pitch
        count = len(y) // 160 + 1
        f0 = np.zeros(count, np.float32)
        for first in range(0, count, 3000):
            last = min(count, first + 3000)
            left, right = max(0, first - 100), min(count, last + 100)
            estimate = self.rmvpe.infer_from_audio(y[left * 160:min(len(y), right * 160)], thred=.03)
            core = estimate[first - left:last - left]
            f0[first:first + len(core)] = core
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=160)[0]
        n = min(len(rms), len(f0))
        np.savez_compressed(target, f0=f0[:n], rms=rms[:n], hop_seconds=.01)
        return load_pitch(target)

    def compare(self, source, converted, references, out):
        from ..pitch_validation import compare_conversion
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        x, y = read_at(source, 16000), read_at(converted, 16000)
        alignment = estimate_alignment(x, y, 16000)
        y = shift_audio(y, alignment['applied_shift_samples'], len(x))
        pitch = compare_conversion(self.pitch(x.mean(axis=1), out / 'source_f0.npz'),
                                   self.pitch(y.mean(axis=1), out / 'converted_f0.npz'))
        if not alignment['consistent']: pitch['status'] = 'unassessable'
        reference_vectors = [self.embedding(p) for p in references]
        scores = [float(self.embedding(converted) @ v) for v in reference_vectors]
        result = {'speaker_similarity': float(np.mean(scores)), 'speaker_similarity_per_reference': scores,
                  'source_speaker_similarity': float(self.embedding(source) @ self.embedding(converted)),
                  'speaker_model': 'speechbrain/spkrec-ecapa-voxceleb; raw normalized embeddings',
                  'pitch': pitch, 'alignment': alignment, 'audio': inspect_audio(converted),
                  'source_sha256': sha256(source), 'target_reference_sha256': [sha256(p) for p in references],
                  'subjective_quality': 'awaiting_human_review', 'lyrics_accuracy': 'requires_transcript_and_human_review',
                  'note': 'Speech-trained embeddings are diagnostic for singing; scores are not identity percentages.'}
        write_json(out / 'metrics.json', result)
        return result


def evaluate_voice(request, output):
    result = VoiceMetrics().compare(request['source'], request['converted'], [request['reference']], Path(output) / 'voice_diagnostics')
    result['reference_independence'] = 'same_inference_prompt; diagnostic_only'
    if request.get('conversion'): result['conversion'] = json.loads(Path(request['conversion']).read_text())
    write_json(Path(output) / 'voice_conversion_evaluation.json', result)
