#!/usr/bin/env python3
"""Create a voice-comparison run referencing verified, immutable 00–04 outputs."""
import argparse
import copy
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from music_cover.audio import sha256
from music_cover.common.io import write_json
from music_cover.modules import IDS


def fork(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Destination already exists; source is never overwritten')
    state=json.loads((source/'stage_state.json').read_text())
    if state.get('schema_version')!=2:raise ValueError('Require a modular source run')
    kept={}
    for name in IDS[:5]:
        record=state['stages'][name]
        if record['status']!='succeeded':raise ValueError('Source stage incomplete: '+name)
        marker=Path(record['attempt'])/'outputs.json'
        if sha256(marker)!=record['outputs_manifest_sha256']:raise ValueError('Source stage manifest changed')
        for item in record['outputs'].values():
            file=Path(item['path'])
            if not file.is_file() or file.stat().st_size!=item['bytes'] or sha256(file)!=item['sha256']:
                raise ValueError('Source artifact changed: '+str(file))
        kept[name]=copy.deepcopy(record)
    destination.mkdir(parents=True)
    for name in ('request.json','config.resolved.json'):write_json(destination/name,json.loads((source/name).read_text()))
    write_json(destination/'stage_state.json',{'schema_version':2,'stages':kept})
    write_json(destination/'shared_upstream.json',{'source_run':str(source),'stages':list(kept),
        'policy':'00–04 files are referenced and hash-checked again by the pipeline; retain source run while rerunning dependent stages'})
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    print(fork(ROOT/a.source_run,ROOT/a.run))

if __name__=='__main__':main()
