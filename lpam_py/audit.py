"""Recheck all 214 attempts, frozen inputs/core versions and final-only assets."""
import json
from pathlib import Path
import sys
from PIL import Image
from .run import ROOT,MANAGER,DEFAULTS,parser,load_pairs,implementation,pair_key,record_path,json_read,json_write,digest


def main():
    sys.path.insert(0,str(MANAGER/'tools'))
    from finalize_lpam import audit
    sources,version=implementation()
    for name,expected in sources['native']['sources'].items():
        if digest(ROOT/name)!=expected: raise RuntimeError('Original-kernel/adapter source changed: '+name)
    totals=dict(attempts=0,final_images=0,metric_successes=0)
    for benchmark,(root,manifest,expected) in DEFAULTS.items():
        args=parser().parse_args(['--benchmark',benchmark]); args.result_root=root; args.manifest=manifest
        pairs=load_pairs(args)
        metadata_path=root/'run_metadata.json' if benchmark=='stitchbench' else root/'_global_work/lpam/run_metadata.json'
        metadata=json_read(metadata_path)
        run=metadata['supplemental_methods']['lpam'] if benchmark=='stitchbench' else metadata
        records=run['records'] if benchmark=='stitchbench' else {p['pair_name']:json_read(record_path(p,args)) for p in pairs}
        for pair in pairs:
            record=records[pair['pair_name']]
            key,config=pair_key(pair,args,version)
            if record['cache_key']!=key or record['configuration']!=config:
                raise RuntimeError('Input/parameter/implementation version mismatch: '+pair['pair_name'])
            if record['inference_status']=='success':
                info=record['render_info']
                with Image.open(record['raw_path']) as image:
                    if image.size!=(info['canvas_width'],info['canvas_height']): raise RuntimeError('Canvas dimensions changed')
                    if image.height>8000 or image.width*image.height>16000000: raise RuntimeError('Canvas limit violated')
                if info['max_input_edge'] not in (2048,1536,1024): raise RuntimeError('Unexpected input scale')
                if info['max_input_edge']<2048 and (not info['attempts'] or
                    any(not item['error'].startswith(('MemoryError:','CanvasLimit:')) for item in info['attempts'])):
                    raise RuntimeError('Non-resource scale fallback')
        if benchmark!='stitchbench':
            allowed={'records','run_metadata.json','audit.json','per_pair_metrics.csv','summary_all.csv','failures.csv'}
            if {p.name for p in (root/'_global_work/lpam').iterdir()}-allowed: raise RuntimeError('Unexpected centralized assets')
            for path in (root/'_global_work/lpam/records').rglob('*'):
                if path.is_file() and path.name not in ('metrics.json','method_status.json'):
                    raise RuntimeError('Unexpected per-pair record asset: '+str(path))
        else:
            panels=list((root/'visual_compare').glob('*.jpg'))
            if len(panels)!=20: raise RuntimeError('Expected 20 comparison panels')
            for panel in panels:
                with Image.open(panel) as image:
                    if image.size!=(1840,1374): raise RuntimeError('Comparison layout is not four columns / 13 entries')
                    image.verify()
        from .run import csv_read
        audit(root,csv_read(manifest),records,run,benchmark)
        json_write(metadata_path,metadata)
        for field in totals: totals[field]+=run['audit'][field]
    if totals['attempts']!=214: raise RuntimeError('Incomplete total attempt coverage')
    print('Combined LPAM audit passed: '+json.dumps(totals),flush=True)


if __name__=='__main__': main()
