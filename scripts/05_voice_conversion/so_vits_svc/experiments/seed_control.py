import argparse,json,os,subprocess
from pathlib import Path
from music_cover.common.paths import ROOT
from music_cover.audio import sha256
p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--cases');p.add_argument('--output');a=p.parse_args()
exp=ROOT/'runs/svc_optimization_yang_kun_20261028'
cases=json.loads((exp/'screening_initial_request.json').read_text())['cases']
if a.full:cases=[{'id':c['id'],'source':c['source']} for c in json.loads((exp/'cases.json').read_text())]
if a.cases:cases=json.loads(Path(a.cases).read_text())
out=(Path(a.output) if a.output else exp/('seed_full' if a.full else 'seed_selection')).resolve();out.mkdir(exist_ok=True)
profile=ROOT/'models/voices/yang_kun/seed_vc/catalog_v001_20261027/voice.json';before=sha256(profile)
rows=[]
for c in cases:
 folder=out/'seed_1000/default'/c['id'];folder.mkdir(parents=True,exist_ok=True)
 cmd=[str(ROOT/'envs/seed-vc/bin/python'),'-m','music_cover.backends.cli','convert','--input',c['source'],'--reference',str(ROOT/'data/raw/references/yang_kun/reference.wav'),'--output',str(folder),'--voice-mode','finetuned','--voice-profile',str(profile),'--steps','30','--seed','42']
 if not (folder/'conversion.json').exists():
  with (folder/'process.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 if not (folder/'converted.wav').is_file():raise RuntimeError('Seed output missing: '+str(folder))
 rows.append({'model':'seed_1000','variant':'default','case':c['id'],'source':c['source'],'output':str(folder/'converted.wav'),'status':'succeeded','role':c.get('role','conversion')})
 (out/'outputs.json').write_text(json.dumps(rows,indent=2));print(c['id'],'done',flush=True)
assert sha256(profile)==before
