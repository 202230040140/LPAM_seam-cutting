"""SIFT matching and the original normalized-coordinate homography pipeline."""
import cv2
import numpy as np


class CanvasLimit(RuntimeError): pass


def normalize(points):
    center=points.mean(axis=0); distances=np.linalg.norm(points-center,axis=1)
    if distances.mean()<1e-10: raise ValueError('Degenerate feature coordinates')
    s=np.sqrt(2.)/distances.mean()
    t=np.array([[s,0,-s*center[0]],[0,s,-s*center[1]],[0,0,1.]])
    return (points-center)*s,t


def dlt(a,b):
    n=len(a); matrix=np.zeros((2*n,9))
    x,y=a.T; u,v=b.T
    matrix[0::2]=np.column_stack((x,y,np.ones(n),np.zeros((n,3)),-x*u,-y*u,-u))
    matrix[1::2]=np.column_stack((np.zeros((n,3)),x,y,np.ones(n),-x*v,-y*v,-v))
    _,_,vh=np.linalg.svd(matrix,full_matrices=n==4)
    h=vh[-1].reshape(3,3)
    return h


def estimate(a,b):
    # Coordinates are one-based, as in MATLAB Location and homogeneous DLT.
    original_a,original_b=a.copy(),b.copy()
    a,ta=normalize(a); b,tb=normalize(b)
    rng=np.random.RandomState(0); best=np.zeros(len(a),bool); trials=2000
    for i in range(2000):
        if i>=trials: break
        ids=rng.choice(len(a),4,replace=False)
        if np.linalg.matrix_rank(np.column_stack((a[ids],np.ones(4))))<3: continue
        if np.linalg.matrix_rank(np.column_stack((b[ids],np.ones(4))))<3: continue
        h=dlt(a[ids],b[ids]); p=np.column_stack((a,np.ones(len(a))))@h.T
        with np.errstate(divide='ignore',invalid='ignore'):
            error=np.linalg.norm(p[:,:2]/p[:,2:]-b,axis=1)
        good=error<0.01
        if good.sum()>best.sum():
            best=good
            ratio=float(good.mean())
            if ratio==1: trials=i+1
            elif ratio>0: trials=min(trials,max(i+1,int(np.ceil(np.log(.01)/np.log(1-ratio**4)))))
    if best.sum()<4: raise RuntimeError('Fewer than four normalized RANSAC inliers')
    # homoRANSAC removes duplicate inlier coordinates a second time.
    ia,ib=original_a[best],original_b[best]
    _,ua=np.unique(ia,axis=0,return_index=True); _,ub=np.unique(ib,axis=0,return_index=True)
    ids=np.intersect1d(ua,ub); ia,ib=ia[ids],ib[ids]
    if len(ia)<4: raise RuntimeError('Fewer than four unique RANSAC inliers')
    # calcHomo -> normalise2dpts -> homography_fit -> vgg_H_from_x_lin.
    # VGG applies anisotropic sample-standard-deviation conditioning internally.
    ia,ta=normalize(ia); ib,tb=normalize(ib)
    def condition(points):
        center=points.mean(axis=0); std=points.std(axis=0,ddof=1)
        std=std+(std==0); scale=np.sqrt(2.)/std
        t=np.array([[scale[0],0,-scale[0]*center[0]],
                    [0,scale[1],-scale[1]*center[1]],[0,0,1.]])
        return (points-center)*scale,t
    ia,ca=condition(ia); ib,cb=condition(ib)
    h=np.linalg.solve(tb,np.linalg.solve(cb,dlt(ia,ib)@ca)@ta)
    if abs(h[2,2])>1e-12: h/=h[2,2]
    return h,len(ia)


def match(a,b):
    detector=cv2.SIFT_create()
    ka,da=detector.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),None)
    kb,db=detector.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),None)
    if da is None or db is None or min(len(da),len(db))<4: raise RuntimeError('Insufficient SIFT features')
    da=da/np.maximum(np.linalg.norm(da,axis=1,keepdims=True),1e-12)
    db=db/np.maximum(np.linalg.norm(db,axis=1,keepdims=True),1e-12)
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(da.astype(np.float32),db.astype(np.float32),k=2)
    selected=[m for pair in pairs if len(pair)==2 for m,n in [pair]
              if m.distance**2<.4 and m.distance**2<.6*n.distance**2]
    pa=np.array([ka[m.queryIdx].pt for m in selected],float).reshape(-1,2)+1
    pb=np.array([kb[m.trainIdx].pt for m in selected],float).reshape(-1,2)+1
    # MATLAB keeps the intersection of the first unique coordinate occurrences.
    _,ia=np.unique(pa,axis=0,return_index=True); _,ib=np.unique(pb,axis=0,return_index=True)
    ids=np.intersect1d(ia,ib); pa,pb=pa[ids],pb[ids]
    if len(pa)<4: raise RuntimeError('Insufficient unique SIFT matches')
    return pa,pb,dict(keypoints1=len(ka),keypoints2=len(kb),matches=len(pa))


def align(a,b,max_height=8000,max_pixels=16000000):
    pa,pb,info=match(a,b); h,inliers=estimate(pa,pb); info['inliers']=inliers
    shift1=np.array([[1,0,1],[0,1,1],[0,0,1.]])
    h0=np.linalg.inv(shift1)@h@shift1
    ah,aw=a.shape[:2]; bh,bw=b.shape[:2]
    pts=np.array([[0,0,1],[aw-1,0,1],[0,ah-1,1],[aw-1,ah-1,1.]])@h0.T
    if np.any(np.abs(pts[:,2])<1e-8) or (np.min(pts[:,2])<0<np.max(pts[:,2])):
        raise RuntimeError('Projective horizon crosses target image')
    pts=pts[:,:2]/pts[:,2:]
    if not np.isfinite(pts).all(): raise RuntimeError('Non-finite projected corners')
    xmin,ymin=np.floor(np.minimum(pts.min(axis=0),[0,0])).astype(np.int64)
    xmax,ymax=np.ceil(np.maximum(pts.max(axis=0),[bw-1,bh-1])).astype(np.int64)
    width,height=int(xmax-xmin+3),int(ymax-ymin+3)
    if height>max_height or width*height>max_pixels or min(width,height)<1:
        raise CanvasLimit(f'Canvas {width}x{height} exceeds limits')
    translation=np.array([[1,0,1-xmin],[0,1,1-ymin],[0,0,1.]])
    wa=cv2.warpPerspective(a.astype(np.float64)/255,translation@h0,(width,height),flags=cv2.INTER_LINEAR)
    wb=np.zeros_like(wa); x,y=int(1-xmin),int(1-ymin)
    wb[y:y+bh,x:x+bw]=b.astype(np.float64)/255
    info.update(canvas_width=width,canvas_height=height,homography=h.tolist())
    return wa,wb,info
