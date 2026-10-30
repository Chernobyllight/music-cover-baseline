"""Download/verify isolated optional assets against the committed SHA256 lock."""
import argparse,json,hashlib,urllib.request,zipfile
from pathlib import Path
from music_cover.common.paths import ROOT
p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args()
lock=json.loads((ROOT/'configs/so_vits_svc_experiments_assets.lock.json').read_text())
folder=ROOT/'models/SoVitsSVC/optimization_20261028';folder.mkdir(parents=True,exist_ok=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
report=[]
for item in lock['assets']:
 f=folder/item['name']
 if not f.exists() and a.download:
  temp=f.with_suffix(f.suffix+'.part')
  with urllib.request.urlopen(urllib.request.Request(item['url'],headers={'User-Agent':'Mozilla/5.0'}),timeout=120) as response,temp.open('wb') as out:
   for b in iter(lambda:response.read(4*1024*1024),b''):out.write(b)
  if temp.stat().st_size!=item['bytes'] or digest(temp)!=item['sha256']:raise ValueError('Downloaded checksum mismatch: '+item['name'])
  temp.replace(f)
 if not f.is_file() or f.stat().st_size!=item['bytes'] or digest(f)!=item['sha256']:raise ValueError('Asset missing/corrupt: '+str(f))
 if f.suffix=='.zip' and a.download:
  with zipfile.ZipFile(f) as z:
   if any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):raise ValueError('Unsafe zip member')
   z.extractall(folder/'nsf_hifigan')
 report.append({'name':item['name'],'sha256':item['sha256'],'verified':True});print(item['name'],'verified',flush=True)
(folder/'verification.json').write_text(json.dumps(report,indent=2))
