"""CPU preparation and one BS-RoFormer block for the shared MPS profiler."""
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from audio_separator.separator import Separator
from audio_separator.separator.uvr_lib_v5 import spec_utils

ROOT = Path(__file__).resolve().parents[2]
START = 40 * 44100
LENGTH = 8 * 44100
MODEL = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'


def load(device, output):
    sep = Separator(info_only=device == 'cpu', log_level=logging.WARNING,
                    output_dir=str(output), model_file_dir=str(ROOT / 'models/separation'),
                    normalization_threshold=1.0, amplification_threshold=0.0,
                    sample_rate=44100, use_soundfile=True, output_format='WAV')
    if device == 'cpu':
        sep.torch_device = sep.torch_device_cpu = torch.device('cpu')
    if sep.torch_device.type != device or sep.use_autocast:
        raise RuntimeError('Unexpected separator device or precision')
    sep.load_model(model_filename=MODEL)
    return sep.model_instance


def prepare(run, output):
    import torchvision, torchaudio, onnx2torch
    output.mkdir(exist_ok=True, parents=True)
    model = load('cpu', output)
    path = Path(json.loads((run/'stage_state.json').read_text())['stages']['03_separate']['outputs']['vocals']['path'])
    mix = model.prepare_mix(str(path.with_name('separation_input.wav')))
    mix = spec_utils.normalize(wave=mix, max_peak=1.0, min_peak=0.0)
    block = mix[:, START:START+LENGTH].copy()
    cfg = model.model_data_cfgdict
    assert LENGTH == cfg.audio.hop_length * (cfg.inference.dim_t - 1)
    assert cfg.audio.sample_rate == 44100 and model.overlap == 8 and not model.override_model_segment_size
    assert block.shape == (2, LENGTH) and block.dtype == np.float32
    dest = output/'separator-block.npy'
    if dest.exists() and not np.array_equal(np.load(dest), block):
        raise RuntimeError('Candidate preprocessing differs from the baseline')
    np.save(dest, block)
    print('CPU weights and preprocessing OK', torch.__version__, torchaudio.__version__, torchvision.__version__,
          hashlib.sha256(block.tobytes()).hexdigest())


def run_block(run, output):
    report = {'start_sample': START, 'sample_rate': 44100, 'torch': torch.__version__}
    block = np.load(output.parent/'separator-block.npy')
    report['input_sha256'] = hashlib.sha256(block.tobytes()).hexdigest()
    start = time.perf_counter()
    instance = load('mps', output)
    model = instance.model_run
    tensor = torch.from_numpy(block)[None].to('mps')
    torch.mps.synchronize()
    report['load_and_input_s'] = time.perf_counter() - start
    assert next(model.parameters()).device.type == tensor.device.type == 'mps'
    original = torch.nn.functional.scaled_dot_product_attention
    calls = {}
    def record(q, k, v, *args, **kwargs):
        key = str((tuple(q.shape), tuple(k.shape), str(q.dtype), str(q.device)))
        calls[key] = calls.get(key, 0) + 1
        return original(q, k, v, *args, **kwargs)
    with patch.object(torch.nn.functional, 'scaled_dot_product_attention', record):
        for phase in ('first', 'second'):
            start = time.perf_counter()
            pred = model(tensor)
            torch.mps.synchronize()
            report[phase+'_s'] = time.perf_counter()-start
            if phase == 'first':
                vocals = pred[0].reshape(2, LENGTH).cpu().numpy()
                assert calls and np.isfinite(vocals).all()
                for name, audio in [('vocals', vocals), ('instrumental', block-vocals)]:
                    sf.write(output/(name+'.wav'), audio.T, 44100, subtype='FLOAT')
                report['sdpa_first'] = dict(calls)
            (output/'block.json').write_text(json.dumps(report, indent=2)+'\n')
            if float(os.environ['MUSIC_PROBE_END']) - time.monotonic() < report['first_s']*1.3+2:
                break


if __name__ == '__main__':
    import sys
    prepare(*(Path(s) for s in sys.argv[1:]))
