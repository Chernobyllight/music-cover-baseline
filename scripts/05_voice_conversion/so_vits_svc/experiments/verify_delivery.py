import json
from pathlib import Path
import numpy as np,soundfile as sf
from music_cover.common.paths import ROOT
from music_cover.audio import sha256
r=ROOT/'runs/svc_optimization_yang_kun_20261028';run=ROOT/'runs/lhd_yang_kun_svc_optimized_validation_20261028'
state=json.loads((run/'stage_state.json').read_text());assert len(state['stages'])==12 and all(s['status']=='succeeded' for s in state['stages'].values())
prod=Path(state['stages']['05_voice_conversion']['outputs']['converted']['path']);experiment=r/'optimized_full/low_lr_refine_2000/chunk8/hero_celtic/converted.wav'
a,sr=sf.read(prod,dtype='float32');b,srb=sf.read(experiment,dtype='float32');assert sr==srb and a.shape==b.shape
error=float(np.max(np.abs(a-b)));rms_error=float(np.sqrt(np.mean((a-b)**2)))
parity=json.loads((r/'parity_probe/result.json').read_text());assert parity['all_exported_model_tensors_equal'] and parity['max_abs_error']<1e-5
assert sha256(run/'stages/05_voice_conversion/attempt_001/artifacts/source_mono.wav')==sha256(r/'cases/hero_celtic/source.wav')
checks=json.loads((run/'deliverables/delivery_validation.json').read_text());assert checks['technical_validation']=='passed'
report={'all_12_stages_succeeded':True,'shared_stages':'00–04 reused and hash-checked','production_vs_experiment_max_abs_error':error,'production_vs_experiment_rms_error':rms_error,'parity_probe':parity,'numerical_note':'Separate default GPU processes are not bitwise identical; equal exported tensors and same-process deterministic-CuDNN comparison are exact.','converted_sha256':sha256(prod),'delivery_validation':checks,'seed_regression':json.loads((r/'regression/seed_regression.json').read_text())}
for folder in ['baseline_full','optimized_full','seed_full']:
 d=json.loads((r/folder/'deliveries.json').read_text());assert len(d)==4
 for x in d:
  assert x['mix']['final']['rms']>1e-5 and x['mix']['final']['sample_peak']<=1.001
  assert x['mix']['final']['estimated_true_peak_dbtp']<=-.9
report['twelve_mixes_passed']=True
(r/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['delivery_validation','seed_regression']},indent=2))
