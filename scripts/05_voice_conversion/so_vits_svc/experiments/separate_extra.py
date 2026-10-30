import json,os,logging,time
from pathlib import Path
import torch
from audio_separator.separator import Separator
from music_cover.common.paths import ROOT
from music_cover.audio import sha256
torch.set_num_threads(4)
out=ROOT/'data/voice_datasets/yang_kun/so_vits_svc/optimization_20261028/extra_kongcheng';out.mkdir(parents=True,exist_ok=True)
source=ROOT/'data/raw/songs/kong_cheng_yang_kun.mp3';original=source
for name,model,names in [('vocals','model_bs_roformer_ep_317_sdr_12.9755.ckpt',{'Vocals':'vocals','Instrumental':'instrumental'}),('lead','mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',{'Vocals':'lead','Instrumental':'backing'})]:
 if not (out/(name+'.wav')).exists():
  s=Separator(model_file_dir=str(ROOT/'models/separation'),output_dir=str(out),output_format='WAV',normalization_threshold=1.,amplification_threshold=0.,sample_rate=44100,use_soundfile=True,log_level=logging.WARNING,mdxc_params={'segment_size':256,'override_model_segment_size':False,'batch_size':1,'overlap':4,'pitch_shift':0})
  s.load_model(model_filename=model);os.chdir(out);start=time.monotonic();s.separate(str(source),names);del s;torch.cuda.empty_cache()
  print(name,time.monotonic()-start,flush=True)
 source=out/'vocals.wav'
(out/'source.json').write_text(json.dumps({'path':str(original),'sha256':sha256(original),'metadata':str(original.with_suffix('.source.json')),'song_group':'yang_kun:空城','split':'train','note':'Additional full recording from existing source library; not validation/test'},ensure_ascii=False,indent=2))
