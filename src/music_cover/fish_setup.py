"""Explicit Fish asset deployment and read-only preflight; never installs packages."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from .audio import sha256
from .common.paths import ROOT
from .common.io import write_json
from .fish import check_assets, UPSTREAM_COMMIT


def apply_compatibility(upstream):
    entries=json.loads((ROOT/'patches/fish_diffusion/files.json').read_text())
    actual={name:sha256(upstream/name) for name in entries}
    if all(actual[name]==row['patched_sha256'] for name,row in entries.items()):return
    if not all(actual[name]==row['original_sha256'] for name,row in entries.items()):
        raise ValueError('Fish source differs from the pinned original or complete compatibility patch')
    subprocess.run(['git','apply','--check',str(ROOT/'patches/fish_diffusion/base.patch')],cwd=upstream,check=True)
    subprocess.run(['git','apply',str(ROOT/'patches/fish_diffusion/base.patch')],cwd=upstream,check=True)


def download(cache_dir=None):
    lock=json.loads((ROOT/'configs/fish_diffusion_assets.lock.json').read_text())
    upstream=ROOT/'third_party/fish-diffusion'
    if not upstream.exists():
        subprocess.run(['git','clone',lock['upstream_url'],str(upstream)],check=True)
        subprocess.run(['git','checkout',UPSTREAM_COMMIT],cwd=upstream,check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()
    if revision!=UPSTREAM_COMMIT:raise ValueError('Fish checkout is a different revision; refusing to reset it')
    apply_compatibility(upstream)
    cache=(ROOT/Path(cache_dir).expanduser()).resolve() if cache_dir else ROOT/'models/FishDiffusion/downloads'
    cache.mkdir(parents=True,exist_ok=True)
    for item in lock['assets']:
        target=ROOT/item['path']
        if target.is_file() and target.stat().st_size==item['bytes'] and sha256(target)==item['sha256']:continue
        name=item['url'].rsplit('/',1)[-1];archive=cache/name
        if not archive.is_file():
            partial=archive.with_suffix(archive.suffix+'.partial')
            # Transfers are explicit and bounded. An offline cache may be supplied.
            subprocess.run(['curl','-fL','--retry','3','--connect-timeout','20','--max-time','1800',
                            '-o',str(partial),item['url']],check=True)
            partial.replace(archive)
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.partial')
        if 'archive_member' in item:
            with zipfile.ZipFile(archive) as z, z.open(item['archive_member']) as inp, temporary.open('wb') as out:
                shutil.copyfileobj(inp,out)
        else:shutil.copy2(archive,temporary)
        if temporary.stat().st_size!=item['bytes'] or sha256(temporary)!=item['sha256']:
            temporary.unlink();raise ValueError('Downloaded Fish asset checksum differs: '+name)
        temporary.replace(target)
    check_assets(training=True)
    print('Fish source, compatibility patch and weights verified.')


def check(scope='inference',model_load=False,gpu=0,output=None):
    if Path(sys.prefix).resolve()!=Path('/opt/conda').resolve():raise RuntimeError('Use base /opt/conda/bin/python')
    from .common.resources import preflight
    preflight(ROOT,['fish_diffusion','voice_metrics'],mp3=True)
    check_assets(training=scope=='training')
    report={'python':sys.executable,'prefix':sys.prefix,'scope':scope,'assets':'verified','upstream_commit':UPSTREAM_COMMIT}
    if model_load:
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
        from .backends.fish_diffusion import build_model,FeatureExtractor
        checkpoint=ROOT/'models/FishDiffusion/content-vec-pretrained-v1.ckpt'
        model,coverage=build_model(checkpoint,pretrained=True)
        extractor=FeatureExtractor()
        report.update(parameter_loading=coverage,trainable_parameters=sum(p.numel() for p in model.parameters()),
                      content_layer=extractor.encoder.output_layer,content_projection=extractor.encoder.use_projection)
    print(json.dumps(report,indent=2))
    if output:write_json(ROOT/output,report)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    d=sub.add_parser('download');d.add_argument('--cache-dir')
    c=sub.add_parser('check');c.add_argument('--scope',choices=['training','inference'],default='inference')
    c.add_argument('--model-load',action='store_true');c.add_argument('--gpu',type=int,default=0);c.add_argument('--output')
    a=p.parse_args()
    if a.operation=='download':download(a.cache_dir)
    else:check(a.scope,a.model_load,a.gpu,a.output)


if __name__=='__main__':main()
