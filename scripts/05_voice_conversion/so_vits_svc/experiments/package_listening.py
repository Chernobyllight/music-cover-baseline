import json,tarfile,hashlib
from pathlib import Path
from music_cover.common.paths import ROOT
r=ROOT/'runs/svc_optimization_yang_kun_20261028';entries=[]
methods={'baseline_full':'svc_original','optimized_full':'svc_optimized','seed_full':'seed_1000'}
archive=r/'listening_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for folder,method in methods.items():
  rows=json.loads((r/folder/'deliveries.json').read_text());assert len(rows)==4
  for x in rows:
   for key,name in [('final_mp3','final.mp3'),('vocal_mp3','vocal.mp3')]:
    p=Path(x[key]);target=f'audio/{x["case"]}/{method}/{name}';tar.add(p,arcname=target)
    entries.append({'path':target,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'server_path':str(p)})
 for name in ['summary.json','selection.json','contentvec_equivalence.json']:
  tar.add(r/name,arcname=name)
 for name in ['baseline_full','optimized_full','seed_full','final_validation_seed']:
  tar.add(r/name/'evaluation/results.json',arcname=f'metrics/{name}.json')
manifest=r/'listening_manifest.json';manifest.write_text(json.dumps(entries,indent=2));print(archive,archive.stat().st_size)
