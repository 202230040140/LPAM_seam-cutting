"""One isolated inference process. No persistent intermediate images."""
import argparse
import json
from pathlib import Path
import time
import cv2
from .geometry import align,CanvasLimit
from .core import seam_cut,improve,composite


def stitch(left,right,output,max_edge=2048,max_height=8000,max_pixels=16000000):
    original=[]
    for path in (left,right):
        image=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if image is None: raise IOError(f'Cannot read input {path}')
        original.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    attempts=[]; started=time.perf_counter()
    for edge in dict.fromkeys([max_edge]+[e for e in (1536,1024) if e<max_edge]):
        phase='input_scaling'
        try:
            scale=min(1.,edge/max(max(im.shape[:2]) for im in original))
            images=[cv2.resize(im,(max(1,round(im.shape[1]*scale)),max(1,round(im.shape[0]*scale))),interpolation=cv2.INTER_AREA) if scale<1 else im for im in original]
            print(f'align edge={edge}',flush=True)
            phase='initial_homography'
            a,b,info=align(*images,max_height,max_pixels)
            print('initial graph cut',flush=True)
            phase='initial_graph_cut'
            sa,sb,energy=seam_cut(a,b)
            print('LPAM local alignment',flush=True)
            phase='lpam_local_repair'
            wa,wb,la,lb,lpam=improve(a,b,sa,sb)
            result=composite(wa,wb,la,lb)
            phase='final_composition'
            if not cv2.imwrite(str(output),cv2.cvtColor(result,cv2.COLOR_RGB2BGR)): raise IOError('Cannot save final panorama')
            return dict(**info,**lpam,initial_cut_energy=energy,input_scale=scale,max_input_edge=edge,
                        input_shapes=[list(im.shape[:2]) for im in images],attempts=attempts,
                        inference_seconds=time.perf_counter()-started)
        except (MemoryError,CanvasLimit) as exc:
            attempts.append(dict(edge=edge,error=f'{type(exc).__name__}: {exc}'))
            if edge==1024 or edge==min([max_edge,1024]):
                exc.lpam_phase=phase; exc.lpam_attempts=attempts; raise
        except Exception as exc:
            exc.lpam_phase=phase; exc.lpam_attempts=attempts; raise
    raise RuntimeError(f'All resource retries failed: {attempts}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args(); request=json.loads(args.request.read_text(encoding='utf-8'))
    try:
        info=stitch(request['left'],request['right'],request['output'],request['max_input_edge'],
                    request['max_out_height'],request['max_canvas_pixels'])
        payload=dict(success=True,**info)
    except Exception as exc:
        payload=dict(success=False,failure_reason=f'{type(exc).__name__}: {exc}',
                     failure_stage=getattr(exc,'lpam_phase','input_read'),attempts=getattr(exc,'lpam_attempts',[]))
    Path(request['status']).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    if not payload['success']: raise SystemExit(payload['failure_reason'])


if __name__=='__main__': main()
