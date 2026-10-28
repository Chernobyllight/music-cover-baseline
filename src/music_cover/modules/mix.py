from ..audio import mix_tracks, read_at, fit_length
from ..common.io import write_json
import json
import shutil
import subprocess
from ..audio import sha256
from ..voice_candidates import contained_file, recommend, render_report
import soundfile as sf
import numpy as np

ID = '06_mix'
CONTRACT_VERSION = 2
DEFAULTS = {'timeout_seconds': 10800, 'mp3_bitrate_kbps': 192}
CODE = ['audio.py', 'voice_candidates.py']
TIMELINE = {'*': 'generated'}
def backends(r, c): return []
def requires(r): return ['03_style_generation', '04_vocal_separation', '05_voice_conversion']
def inputs(ctx):
    result = {'cover': ctx.output('03_style_generation', 'cover'), 'vocals': ctx.output('04_vocal_separation', 'vocals'),
              'instrumental': ctx.output('04_vocal_separation', 'instrumental'), 'converted': ctx.output('05_voice_conversion', 'converted')}
    if 'candidate_report' in ctx.resolved['05_voice_conversion']:
        result.update(ctx.all_outputs('05_voice_conversion', 'candidates/'))
        result['candidate_report'] = ctx.output('05_voice_conversion', 'candidate_report')
    return result
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
    paths = {'final': out / 'final.wav', 'final_mp3': out / 'final.mp3', 'reconstruction': out / 'reconstruction.wav',
            'aligned_vocal': out / 'converted_aligned.wav', 'metrics': out / 'mix.json',
            'vocal_post_gain': out / 'vocal_post_gain.wav', 'instrumental_post_gain': out / 'instrumental_post_gain.wav', 'residual': out / 'mix_residual.wav'}

    if 'candidate_report' in i:
        report = json.loads(i['candidate_report'].read_text())
        selected = recommend(report['candidates'])
        if selected is None or selected['id'] != report['recommended_id']:
            raise ValueError('Candidate recommendation does not match eligible metrics')
        if sha256(i['converted']) != selected['audio_sha256']:
            raise ValueError('Canonical converted audio does not match recommended candidate')
        for row in report['candidates']:
            if not row.get('audio'): continue
            audio = contained_file(i['candidate_report'].parent, row['audio'])
            if sha256(audio) != row['audio_sha256']: raise ValueError('Candidate audio checksum mismatch')
            folder = out / 'candidates' / row['id']; folder.mkdir(parents=True)
            if row['id'] == selected['id']:
                for name in ('final.wav', 'final.mp3', 'mix.json'): shutil.copy2(out / name, folder / name)
            else:
                try:
                    candidate_mix = mix_tracks(i['cover'], i['vocals'], i['instrumental'], audio, folder, mp3_bitrate_kbps=p['mp3_bitrate_kbps'])
                    write_json(folder / 'mix.json', candidate_mix)
                except (ValueError, subprocess.SubprocessError) as error:
                    row.update(mix_status='failed', mix_error=str(error))
                    shutil.rmtree(folder)
                    row['raw_audio'] = row.pop('audio')
                    row.pop('conversion', None)
                    continue
                # Auxiliary alignment files remain available for the selected output at the stage root.
                for name in ('converted_aligned.wav', 'reconstruction.wav'): (folder / name).unlink()
            for kind in ('final_wav', 'final_mp3'):
                file = folder / kind.replace('_', '.')
                row[kind] = str(file.relative_to(out)); row[kind + '_sha256'] = sha256(file)
            row['mix_status'] = 'succeeded'
            row['raw_audio'] = row.pop('audio')
            row.pop('conversion', None)
        report['mix_policy'] = 'same accompaniment and mix algorithm; vocal RMS matching and peak limiting computed per candidate'
        report['metric_audio'] = '05 raw converted vocals, before mixing; not full mixture ECAPA'
        write_json(out / 'candidates.json', report)
        render_report(report, out / 'candidates.html', mixed=True)
        paths['candidate_report'] = out / 'candidates.json'
    return paths
