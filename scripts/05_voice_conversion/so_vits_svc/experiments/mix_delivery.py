"""Mix all comparison versions with frozen accompaniment and the existing 06 algorithm."""
import argparse,json,concurrent.futures
from pathlib import Path
from music_cover.common.paths import ROOT
from music_cover.audio import sha256,mix_tracks,export_mp3
p=argparse.ArgumentParser();p.add_argument('--outputs',required=True);a=p.parse_args()
r=ROOT/'runs/svc_optimization_yang_kun_20261028';rows=json.loads(Path(a.outputs).read_text());cases={c['id']:c for c in json.loads((r/'cases.json').read_text())}
def process(row):
 assert row['status']=='succeeded'
 c=cases[row['case']];assert sha256(c['source'])==c['source_sha256']
 state=json.loads((Path(c['source_run'])/'stage_state.json').read_text())
 def artifact(stage,name):
  x=state['stages'][stage]['outputs'][name];assert sha256(x['path'])==x['sha256'];return x['path']
 cover=artifact('03_style_generation','cover');vocal=artifact('04_vocal_separation','vocals');inst=artifact('04_vocal_separation','instrumental')
 out=r/'delivery'/row['case']/row['model'];out.mkdir(parents=True,exist_ok=True)
 result=mix_tracks(cover,vocal,inst,row['output'],out)
 dry=export_mp3(row['output'],out/'vocal.mp3',192)
 report={**row,'case_split':c['split'],'source_sha256':c['source_sha256'],'instrumental_sha256':sha256(inst),'cover_sha256':sha256(cover),'mix':result,'final_wav':str(out/'final.wav'),'final_mp3':str(out/'final.mp3'),'vocal_mp3':str(out/'vocal.mp3'),'vocal_mp3_info':dry}
 (out/'mix.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:reports=list(pool.map(process,rows))
 (Path(a.outputs).parent/'deliveries.json').write_text(json.dumps(reports,indent=2));print('mixed',len(reports))
