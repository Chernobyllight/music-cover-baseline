import json
from pathlib import Path
import numpy as np,torch,faiss
from music_cover.common.paths import ROOT
exp=ROOT/'runs/svc_optimization_yang_kun_20261028'
folder=ROOT/'data/voice_datasets/yang_kun/so_vits_svc/v001_20261028'
bank=np.concatenate([torch.load(f,map_location='cpu',weights_only=True).squeeze(0).T.numpy() for f in sorted((folder/'train').rglob('*.soft.pt'))]).astype(np.float32)
index=faiss.IndexFlatL2(bank.shape[1]);index.add(bank);faiss.write_index(index,str(exp/'original.index'))
kmeans=faiss.Kmeans(bank.shape[1],64,niter=30,seed=42);kmeans.train(bank);cluster=faiss.IndexFlatL2(bank.shape[1]);cluster.add(kmeans.centroids);faiss.write_index(cluster,str(exp/'original_cluster.index'))
cases=[]
for case in json.loads((exp/'cases.json').read_text()):
 if case['split']=='selection':cases.append({'id':case['id'],'source':case['windows'][1]})
for i,f in enumerate(sorted((ROOT/'data/raw/references/yang_kun/dataset/validation').glob('*.wav'))[::3]):cases.append({'id':'self_'+str(i),'source':str(f),'role':'reconstruction'})
variants=[{'id':'noise_'+str(n),'settings':{'noise':n}} for n in [0,.1,.2,.4,.7]]
variants += [{'id':'f0_'+name,'settings':{'noise':.1,'f0':name}} for name in ['dio','harvest','pm','crepe']]
variants += [{'id':'retrieval_'+str(r),'settings':{'noise':.1,'retrieval':r,'index':str(exp/'original.index')}} for r in [.25,.5,.75]]
variants += [{'id':'cluster_05','settings':{'noise':.1,'retrieval':.5,'index':str(exp/'original_cluster.index')}}]
variants += [{'id':'rms_05','settings':{'noise':.1,'rms_mix':.5}},{'id':'chunk_8','settings':{'noise':.1,'chunk':8.,'overlap':.2}}]
model=ROOT/'models/voices/yang_kun/so_vits_svc/v001_20261028'
req={'output':str(exp/'screening_initial'),'models':[{'id':'svc_1000','checkpoint':str(model/'model.pth'),'config':str(model/'config.json'),'variants':variants}],'cases':cases}
(exp/'screening_initial_request.json').write_text(json.dumps(req,indent=2))
print('cases',len(cases),'variants',len(variants))
