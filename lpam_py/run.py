"""Final-image-only LPAM experiments with isolated CPU inference and shared IQA."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import cv2

ROOT=Path(__file__).resolve().parents[1]
MANAGER=ROOT.parent/'MethodManagement'
sys.path.insert(0,str(ROOT.parent/'UnsupervisedDeepImageStitching/reproduce'))
from hd3d_eval import evaluate_raw,load_niqe_metric,load_lpips_metric

DEFAULTS={
 'stitchbench':(Path(r'D:\StitchBench_Result'),Path(r'D:\StitchBench_Result\manifest.csv'),100),
 'hd3d':(Path(r'D:\HD3D_Result'),Path(r'D:\HD3D_Result\_global_work\_work_root\manifest.csv'),78),
 'lpsd':(Path(r'D:\LPS_Result'),Path(r'D:\LPS_Result\_global_work\_work_root\manifest.csv'),36)}


def now(): return datetime.now(timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict): return {key:clean(val) for key,val in value.items()}
    if isinstance(value,(list,tuple)): return [clean(val) for val in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def json_write(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(payload),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def json_read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig')) if Path(path).is_file() else {}


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle: return list(csv.DictReader(handle))


def evidence(result_root):
    filenames=['per_pair_metrics.csv','summary_all.csv']
    if result_root.name=='StitchBench_Result': filenames+=['by_category.csv','failures.csv']
    result={}
    for filename in filenames:
        path=result_root/filename
        if path.is_file():
            rows=[row for row in csv_read(path) if row.get('method')!='lpam']
            result[filename]=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return result


def load_pairs(args):
    rows=csv_read(args.manifest); pairs=[]
    for row in rows:
        if args.benchmark=='stitchbench':
            scene=row['dataset']; names=row['image_files'].split('|')
            if scene.startswith('NISwGSP-') or len(names)!=2: raise ValueError('Invalid fixed StitchBench scene '+scene)
            pair=dict(scene=scene,pair_id='12',pair_name=scene,left_source=str(Path(row['data_dir'])/names[0]),
                      right_source=str(Path(row['data_dir'])/names[1]),final_pair_dir=str(args.result_root/scene),
                      category=row['category'],split=row['split'],gt_path='')
        else: pair=dict(row)
        target=Path(pair['final_pair_dir']).resolve(); target.relative_to(args.result_root.resolve())
        pair['final_pair_dir']=str(target)
        for key in ('left_source','right_source'):
            if not Path(pair[key]).is_file(): raise FileNotFoundError(pair[key]+': '+pair[key])
        pairs.append(pair)
    if len(pairs)!=DEFAULTS[args.benchmark][2]: raise ValueError(f'Unexpected manifest size {len(pairs)}')
    if len({p['pair_name'] for p in pairs})!=len(pairs): raise ValueError('Duplicate pairs')
    return pairs


def implementation():
    files=sorted((ROOT/'lpam_py').glob('*.py'))+sorted((ROOT/'lpam_py/native_src').glob('*.cpp'))
    # Report/scheduling changes do not change already inferred images.
    files=[p for p in files if p.name not in ('run.py','smoke.py','all.py','audit.py','__init__.py')]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    build=json_read(ROOT/'lpam_py/_native/build_metadata.json')
    if not build: raise RuntimeError('Native kernels not built; run lpam_py.build_native')
    for name,expected in build['dlls'].items():
        if digest(ROOT/'lpam_py/_native'/name)!=expected: raise RuntimeError('Native library hash mismatch')
    hashes['native']=build
    hashes['GT_evaluator']=digest(ROOT.parent/'UnsupervisedDeepImageStitching/reproduce/hd3d_eval.py')
    return hashes,hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def pair_key(pair,args,version):
    configuration=dict(implementation=version,left=digest(pair['left_source']),right=digest(pair['right_source']),
        gt=digest(pair['gt_path']) if pair.get('gt_path') else '',max_input_edge=args.max_input_edge,
        max_out_height=args.max_out_height,max_canvas_pixels=args.max_canvas_pixels,
        lpips_max_side=args.lpips_max_side,benchmark=args.benchmark)
    return hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest(),configuration


def record_path(pair,args):
    return args.result_root/'_global_work/lpam/records'/pair['scene']/Path(pair['final_pair_dir']).name/'metrics.json'


def process(pair,args,version,previous,niqe,lpips,device):
    output=Path(pair['final_pair_dir'])/'lpam/stitch_result.png'
    output.parent.mkdir(parents=True,exist_ok=True)
    key,configuration=pair_key(pair,args,version)
    verified=(previous.get('cache_key')==key and output.is_file() and
              previous.get('output_sha256')==digest(output) and previous.get('inference_status')=='success')
    if args.skip_existing and not args.force and verified and previous.get('evaluation_status')=='success':
        return previous
    result=dict(scene=pair['scene'],pair_id=pair['pair_id'],pair_name=pair['pair_name'],method='lpam',status='failed',
        inference_status='failed',evaluation_status='not_attempted',failure_reason='',cache_key=key,
        configuration=configuration,raw_path=str(output),aligned_path='',valid_mask_path='',gt_path=pair.get('gt_path',''),
        category=pair.get('category',''),split=pair.get('split',''),created_at_utc=now(),mdr_protocol='N/A' if args.benchmark=='stitchbench' else 'gt_alignment')
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='lpam-pair-') as temp:
        temporary=Path(temp)
        try:
            if args.skip_existing and not args.force and verified:
                result.update({field:value for field,value in previous.items() if field in
                               ('render_info','output_sha256','inference_status','inference_seconds')})
            else:
                output.unlink(missing_ok=True)
                request=dict(left=pair['left_source'],right=pair['right_source'],output=str(temporary/'stitch_result.png'),
                    status=str(temporary/'inference.json'),max_input_edge=args.max_input_edge,
                    max_out_height=args.max_out_height,max_canvas_pixels=args.max_canvas_pixels)
                json_write(temporary/'request.json',request)
                # Windows venv launchers spawn a second python.exe. Terminate
                # the entire owned process tree on timeout, not only the stub.
                worker=subprocess.Popen([sys.executable,'-m','lpam_py.worker','--request',str(temporary/'request.json')],
                    cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
                try:
                    worker_output,_=worker.communicate(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    result['failure_stage']='inference_timeout'
                    if os.name=='nt':
                        subprocess.run(['taskkill','/PID',str(worker.pid),'/T','/F'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                    else: worker.kill()
                    worker.communicate()
                    raise
                info=json_read(temporary/'inference.json')
                if not info.get('success'):
                    result['failure_stage']=info.get('failure_stage','inference_process')
                    result['render_info']=info
                    raise RuntimeError(info.get('failure_reason') or f'Inference process exit={worker.returncode}: {worker_output[-1500:]}')
                rendered=temporary/'stitch_result.png'
                image=cv2.imread(str(rendered))
                if image is None: raise RuntimeError('Inference did not emit a valid final image')
                shutil.copy2(rendered,output)
                result.update(inference_status='success',render_info=info,output_sha256=digest(output),inference_seconds=info['inference_seconds'])
            result['evaluation_status']='failed'
            if args.benchmark=='stitchbench':
                import numpy as np
                import torch
                from PIL import Image
                with Image.open(output) as im: array=np.asarray(im.convert('RGB'),np.float32)/255.
                tensor=torch.from_numpy(array).permute(2,0,1).unsqueeze(0).to(device)
                with torch.no_grad(): result['niqe']=float(niqe(tensor).detach().cpu().item())
                if not math.isfinite(result['niqe']): raise RuntimeError('NIQE is non-finite')
                result['mdr']=None
            else:
                metrics=evaluate_raw(output,Path(pair['gt_path']),temporary,niqe,lpips,lpips_max_side=args.lpips_max_side)
                if not all(math.isfinite(float(metrics[name])) for name in ('mdr','niqe','psnr','ssim','lpips','rmse')):
                    raise RuntimeError('GT metric is non-finite')
                result.update(metrics); result.update(aligned_path='',valid_mask_path='')
            result.update(status='success',evaluation_status='success',failure_reason='')
        except Exception as exc:
            result['failure_reason']=f'{type(exc).__name__}: {exc}'
            result.setdefault('failure_stage','evaluation' if result['inference_status']=='success' else 'inference_process')
        finally:
            result['runtime_seconds']=time.perf_counter()-started
            if not output.exists():
                try: output.parent.rmdir()
                except OSError: pass
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except ImportError: pass
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=DEFAULTS,required=True)
    p.add_argument('--manifest',type=Path); p.add_argument('--result-root',type=Path)
    p.add_argument('--scene',action='append'); p.add_argument('--limit',type=int,default=0)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda','auto'])
    p.add_argument('--max-input-edge',type=int,default=2048)
    p.add_argument('--max-out-height',type=int,default=8000)
    p.add_argument('--max-canvas-pixels',type=int,default=16000000)
    p.add_argument('--timeout',type=int,default=600); p.add_argument('--lpips-max-side',type=int,default=1024)
    p.add_argument('--skip-existing',action='store_true'); p.add_argument('--force',action='store_true')
    p.add_argument('--stop-on-error',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if min(args.max_input_edge,args.max_out_height,args.max_canvas_pixels,args.timeout,args.lpips_max_side)<1:
        raise ValueError('Resource bounds must be positive')
    root,manifest,_=DEFAULTS[args.benchmark]
    args.result_root=(args.result_root or root).resolve(); args.manifest=(args.manifest or manifest).resolve()
    all_pairs=load_pairs(args); pairs=all_pairs
    if args.scene: pairs=[p for p in pairs if p['scene'] in args.scene]
    if args.limit>0: pairs=pairs[:args.limit]
    if not pairs: raise ValueError('No selected pairs')
    sources,version=implementation()
    sb=args.benchmark=='stitchbench'
    metadata_path=args.result_root/'run_metadata.json' if sb else args.result_root/'_global_work/lpam/run_metadata.json'
    metadata=json_read(metadata_path)
    run=metadata.setdefault('supplemental_methods',{}).setdefault('lpam',{}) if sb else metadata
    run.setdefault('preserved_tables',evidence(args.result_root))
    if 'preserved_images' not in run:
        run['preserved_images']={str(p.relative_to(args.result_root)):digest(p)
            for p in sorted(args.result_root.rglob('stitch_result.png')) if p.parent.name!='lpam'}
    if sb:
        protected={key:metadata.get(key) for key in ('ddaf','heldout_gate','visual_review')}
        run.setdefault('protected_metadata_sha256',hashlib.sha256(json.dumps(protected,sort_keys=True).encode()).hexdigest())
    run.update(implementation=sources,implementation_sha256=version,configuration='Homography + basic graph cut + LPAM',
        port_notes='Original dense SIFT/BPFlow/maxflow retained; OpenCV initial SIFT/warping, Keys bicubic, NumPy RANSAC. MATLAB numerical identity unverified. Composition uses locally realigned images.',
        fixed_parameters=dict(patch_size=21,k=1.5,beta=8,ransac_seed=0,ransac_max_trials=2000,ransac_normalized_threshold=.01),
        manifest=str(args.manifest),manifest_sha256=digest(args.manifest),updated_at_utc=now())
    run.setdefault('records',{})
    import numpy as np
    import torch
    import importlib.metadata
    run['environment']=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,
        opencv=cv2.__version__,torch=torch.__version__,pyiqa=importlib.metadata.version('pyiqa'),
        cuda_available=torch.cuda.is_available(),inference_device='cpu',requested_evaluation_device=args.device)
    requested='cuda' if args.device=='auto' else args.device
    niqe,device=load_niqe_metric(requested)
    lpips,_=load_lpips_metric(device) if not sb else (None,None)
    for i,pair in enumerate(pairs,1):
        old=run['records'].get(pair['pair_name'],{}) if sb else json_read(record_path(pair,args))
        print(f'[{i}/{len(pairs)}] {args.benchmark} {pair["pair_name"]}',flush=True)
        result=process(pair,args,version,old,niqe,lpips,device)
        if sb:
            run['records'][pair['pair_name']]=result
            metadata.setdefault('collections',{})['lpam']=dict(attempted=len(run['records']),updated_at_utc=now())
        else:
            json_write(record_path(pair,args),result)
            json_write(record_path(pair,args).with_name('method_status.json'),dict(
                success=result['status']=='success',inference_status=result['inference_status'],
                evaluation_status=result['evaluation_status'],failure_reason=result['failure_reason']))
        run['updated_at_utc']=now(); json_write(metadata_path,metadata)
        print(f'  {result["inference_status"]}/{result["evaluation_status"]}; patches={result.get("render_info",{}).get("patch_count","-")}; {result["failure_reason"]}',flush=True)
        if args.stop_on_error and result['status']!='success': return 1
    if not args.scene and not args.limit:
        command=[sys.executable,str(MANAGER/'tools/finalize_lpam.py'),'--benchmark',args.benchmark,
                 '--result-root',str(args.result_root),'--manifest',str(args.manifest)]
        subprocess.run(command,check=True)
    successful=sum((run['records'].get(p['pair_name'],{}) if sb else json_read(record_path(p,args))).get('status')=='success' for p in pairs)
    print(f'Done {args.benchmark}: {successful}/{len(pairs)} success',flush=True)
    return int(successful!=len(pairs))


if __name__=='__main__': raise SystemExit(main())
