from ..audio import mix_tracks, read_at, fit_length
from ..common.io import write_json
import soundfile as sf
import numpy as np

ID = '06_mix'
DEFAULTS = {'timeout_seconds': 10800, 'mp3_bitrate_kbps': 192}
CODE = ['audio.py']
TIMELINE = {'*': 'generated'}
def backends(r, c): return []
def requires(r): return ['03_style_generation', '04_vocal_separation', '05_voice_conversion']
def inputs(ctx): return {'cover': ctx.output('03_style_generation', 'cover'), 'vocals': ctx.output('04_vocal_separation', 'vocals'),
                         'instrumental': ctx.output('04_vocal_separation', 'instrumental'), 'converted': ctx.output('05_voice_conversion', 'converted')}
def params(r, c):
    if c['mp3_bitrate_kbps'] not in {128, 160, 192, 224, 256, 320}:
        raise ValueError('mp3_bitrate_kbps must be 128, 160, 192, 224, 256 or 320')
    return c


def run(ctx, out, i, p):
    result = mix_tracks(i['cover'], i['vocals'], i['instrumental'], i['converted'], out, mp3_bitrate_kbps=p['mp3_bitrate_kbps'])
    write_json(out / 'mix.json', result)
    frames = result['final']['frames']
    gain = result['final']['master_gain']
    vocal = np.repeat(read_at(out / 'converted_aligned.wav'), 2, axis=1) * (gain * result['vocal_gain'])
    instrumental = fit_length(read_at(i['instrumental']), frames) * gain
    sf.write(out / 'vocal_post_gain.wav', vocal, 48000, subtype='FLOAT')
    sf.write(out / 'instrumental_post_gain.wav', instrumental, 48000, subtype='FLOAT')
    sf.write(out / 'mix_residual.wav', read_at(out / 'final.wav') - vocal - instrumental, 48000, subtype='FLOAT')
    return {'final': out / 'final.wav', 'final_mp3': out / 'final.mp3', 'reconstruction': out / 'reconstruction.wav',
            'aligned_vocal': out / 'converted_aligned.wav', 'metrics': out / 'mix.json',
            'vocal_post_gain': out / 'vocal_post_gain.wav', 'instrumental_post_gain': out / 'instrumental_post_gain.wav', 'residual': out / 'mix_residual.wav'}
