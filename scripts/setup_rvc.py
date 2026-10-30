#!/usr/bin/env python3
"""Verify/download pinned RVC assets. Never install or change environment packages."""
import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from music_cover.audio import sha256
from music_cover.rvc import check_assets


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download',action='store_true');p.add_argument('--training',action='store_true');a=p.parse_args()
    lock=json.loads((ROOT/'configs/rvc_assets.lock.json').read_text())
    if a.download:
        for item in lock['assets']:
            if item['scope']=='training' and not a.training:continue
            target=ROOT/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
            if target.is_file() and target.stat().st_size==item['bytes'] and sha256(target)==item['sha256']:continue
            if target.exists():raise ValueError('Existing RVC asset differs; inspect before replacing: '+str(target))
            temporary=target.with_suffix(target.suffix+'.download')
            with urllib.request.urlopen(item['url'],timeout=120) as source,temporary.open('wb') as output:shutil.copyfileobj(source,output)
            if temporary.stat().st_size!=item['bytes'] or sha256(temporary)!=item['sha256']:raise ValueError('RVC download checksum mismatch')
            temporary.replace(target);print('verified:',target)
    check_assets(training=a.training)
    from music_cover.common.resources import preflight
    preflight(ROOT,['rvc']);print('RVC base imports and locked assets OK; no packages changed')


if __name__=='__main__':main()
