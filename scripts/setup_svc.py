#!/usr/bin/env python3
"""Download locked public SVC assets or verify the current installation; never install/downgrade packages."""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from music_cover.audio import sha256
from music_cover.svc import check_assets


def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--training',action='store_true');a=p.parse_args()
    lock=json.loads((ROOT/'configs/so_vits_svc_assets.lock.json').read_text())
    if a.download:
        for item in lock['assets']:
            if item.get('scope')=='training' and not a.training:continue
            target=ROOT/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
            if target.is_file() and target.stat().st_size==item['bytes'] and sha256(target)==item['sha256']:
                print('verified:',target);continue
            if target.exists():raise ValueError('Existing asset differs; inspect it before replacing: '+str(target))
            download=target.with_suffix(target.suffix+'.download')
            with urllib.request.urlopen(item['url'],timeout=120) as response,download.open('wb') as output:shutil.copyfileobj(response,output)
            if item.get('archive_member'):
                extracted=target.with_suffix(target.suffix+'.extracted')
                with zipfile.ZipFile(download) as archive,archive.open(item['archive_member']) as source,extracted.open('wb') as output:shutil.copyfileobj(source,output)
                download.unlink();extracted.rename(download)
            if download.stat().st_size!=item['bytes'] or sha256(download)!=item['sha256']:raise ValueError('Download checksum mismatch: '+str(download))
            download.rename(target);print('downloaded and verified:',target)
    check_assets(training=a.training)
    from music_cover.common.resources import preflight
    preflight(ROOT,['so_vits_svc'])
    print('SVC base environment and locked assets OK')

if __name__=='__main__':main()
