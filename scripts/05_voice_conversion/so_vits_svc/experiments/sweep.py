import argparse,json,traceback
from pathlib import Path
from core import ROOT,EXP,runtime,Converter
import torch
p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
req=json.loads(Path(a.request).read_text());out=Path(req['output']);out.mkdir(parents=True,exist_ok=True)
rows=[]
with runtime(out/'runtime'):
 for model in req['models']:
  converter=Converter(model['checkpoint'],json.loads(Path(model['config']).read_text()))
  for variant in model.get('variants',[{'id':'default','settings':{}}]):
   for case in req['cases']:
    target=out/model['id']/variant['id']/case['id'];target.mkdir(parents=True,exist_ok=True)
    row={'model':model['id'],'variant':variant['id'],'case':case['id'],'source':case['source'],'output':str(target/'converted.wav'),'role':case.get('role','conversion')}
    try:
     if (target/'conversion.json').exists():report=json.loads((target/'conversion.json').read_text())
     else:
      report=converter.infer(case['source'],target/'converted.wav',variant['settings'])
      (target/'conversion.json').write_text(json.dumps(report,indent=2))
     row.update(status='succeeded',report=report)
    except Exception as e:
     row.update(status='failed',error=str(e));(target/'error.txt').write_text(traceback.format_exc())
    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='report'}),flush=True)
    (out/'outputs.json').write_text(json.dumps(rows,indent=2))
  del converter;torch.cuda.empty_cache()
