"""Author's seam selection, local SIFT Flow alignment and seam merging."""
import cv2
import numpy as np
from scipy import ndimage
from .native import dense_sift,discrete_flow,graph_cut


def support(image):
    gray=image@np.array([.2989,.5870,.1140])
    return ndimage.binary_fill_holes(gray>0)


def border(mask):
    return mask & ~ndimage.binary_erosion(mask,structure=ndimage.generate_binary_structure(2,1),border_value=0)


def ids_for(mask):
    flat=mask.ravel(order='F'); ids=np.full(mask.size,-1,np.int32)
    ids[flat]=np.arange(flat.sum(),dtype=np.int32)
    return ids.reshape(mask.shape,order='F')


def hist_otsu(values,edges=None):
    if edges is None:
        # graythresh: 256 discrete levels, threshold normalized to [0,1].
        values=np.floor(np.clip(values,0,1)*255+.5).astype(np.uint8)
        counts=np.bincount(values,minlength=256).astype(float); centers=np.arange(256)/255.
    else:
        counts,_=np.histogram(values,edges); counts=counts.astype(float)
        centers=(edges[:-1]+edges[1:])/2
    if not counts.sum(): return 0.
    p=counts/counts.sum(); w=np.cumsum(p); u=np.cumsum(p*centers)
    with np.errstate(divide='ignore',invalid='ignore'):
        score=(u-u[-1]*w)**2/(w*(1-w))
    finite=np.isfinite(score)
    if not finite.any() or np.nanmax(score)<=0:
        return float(centers[0]+(edges[1]-edges[0])/2) if edges is not None else 0.
    if edges is not None:
        # Original histOstu chooses the first maximum, then adds half a bin.
        idx=np.argmax(np.where(finite,score,-1))
        return float(centers[idx]+(edges[1]-edges[0])/2)
    return float(centers[np.flatnonzero(np.isclose(score,np.nanmax(score),rtol=1e-12))].mean())


def seam_cut(a,b,initial_masks=None):
    if initial_masks is None:
        ma,mb=support(a),support(b); overlap=ma&mb
        if not overlap.any(): raise RuntimeError('Aligned images have no overlap')
        bc=border(overlap); seed_a=border(mb)&bc; seed_b=~seed_a&bc
    else:
        ma=mb=overlap=np.ones(a.shape[:2],bool)
        bc=border(overlap); seed_a=border(initial_masks[0])&bc; seed_b=border(initial_masks[1])&bc
    ids=ids_for(overlap); terminal=np.zeros((int(overlap.sum()),2))
    terminal[ids[seed_a],0]=np.inf; terminal[ids[seed_b],1]=np.inf
    if np.any(seed_a&seed_b): raise RuntimeError('Conflicting local seam boundary labels')
    diff=np.sqrt(np.mean((a-b)**2,axis=2))*overlap
    if initial_masks is not None:
        alpha=hist_otsu(diff[overlap],np.arange(0,1+.00001,.06))
        diff=1/(1+np.exp(-(4/.06)*(diff-alpha)))
    right=overlap[:,:-1]&overlap[:,1:]; down=overlap[:-1,:]&overlap[1:,:]
    def take(array,mask): return array.ravel(order='F')[mask.ravel(order='F')]
    u=np.concatenate((take(ids[:,:-1],right),take(ids[:-1,:],down)))
    v=np.concatenate((take(ids[:,1:],right),take(ids[1:,:],down)))
    cost=np.concatenate((take((diff[:,:-1]+diff[:,1:])/2,right),take((diff[:-1,:]+diff[1:,:])/2,down)))+1e-8
    labels,energy=graph_cut(terminal,u,v,cost)
    assigned=np.zeros(overlap.shape,np.uint8); assigned[overlap]=labels[ids[overlap]]
    sa=ma.copy(); sb=mb.copy(); sa[overlap& (assigned==1)]=False; sb[overlap&(assigned==0)]=False
    return sa,sb,energy


def quality(a,b,overlap,points,size=21):
    # All 21x21 patch SSIM means can be obtained from one valid Gaussian SSIM
    # map and integral sums. This preserves per-channel, overlap-masked SSIM.
    a=a*overlap[:,:,None]; b=b*overlap[:,:,None]
    kernel=cv2.getGaussianKernel(11,1.5)
    def blur(x): return cv2.sepFilter2D(x,-1,kernel,kernel,borderType=cv2.BORDER_REPLICATE)
    ua,ub=blur(a),blur(b); va=blur(a*a)-ua*ua; vb=blur(b*b)-ub*ub; cov=blur(a*b)-ua*ub
    ssim=((2*ua*ub+.01**2)*(2*cov+.03**2))/((ua*ua+ub*ub+.01**2)*(va+vb+.03**2))
    ssim=ssim.mean(axis=2)
    integral=cv2.integral(ssim)
    y,x=points.T; half=(size-1)//2
    y0=np.maximum(y-half,0); y1=np.minimum(y+half+1,a.shape[0])
    x0=np.maximum(x-half,0); x1=np.minimum(x+half+1,a.shape[1])
    iy0=y0+5; iy1=y1-5; ix0=x0+5; ix1=x1-5
    if np.any((iy1<=iy0)|(ix1<=ix0)): raise RuntimeError('Seam quality patch smaller than Gaussian SSIM window')
    means=(integral[iy1,ix1]-integral[iy0,ix1]-integral[iy1,ix0]+integral[iy0,ix0])/((iy1-iy0)*(ix1-ix0))
    return 1-means,np.column_stack((y0,y1,x0,x1))


def cubic(x):
    x=np.abs(x)
    return np.where(x<=1,1.5*x**3-2.5*x**2+1,np.where(x<=2,-.5*x**3+2.5*x**2-4*x+2,0.))


def resize(image,shape):
    """Separable Keys bicubic with MATLAB-style antialiasing and symmetric edges."""
    result=image.astype(np.float64)
    for axis,n in enumerate(shape):
        old=result.shape[axis]
        if old==n: continue
        scale=n/old; width=4/min(scale,1); taps=int(np.ceil(width))+2
        coord=(np.arange(n)+.5)/scale-.5; left=np.floor(coord-width/2).astype(int)
        indices=left[:,None]+np.arange(taps)[None,:]
        dist=coord[:,None]-indices
        weights=min(scale,1)*cubic(dist*min(scale,1)); weights/=weights.sum(axis=1,keepdims=True)
        folded=indices%(2*old); indices=np.where(folded>=old,2*old-1-folded,folded)
        out_shape=list(result.shape); out_shape[axis]=n; output=np.zeros(out_shape,float)
        for k in range(taps):
            block=np.take(result,indices[:,k],axis=axis)
            weight_shape=[1]*result.ndim; weight_shape[axis]=n
            output+=block*weights[:,k].reshape(weight_shape)
        result=output
    if image.dtype==np.uint8: result=np.clip(np.floor(result+.5),0,255).astype(np.uint8)
    return result


def sift_flow(a,b):
    # Same direction as original SIFTflowc2f(sift2,sift1): target is sampled
    # at p+flow(p) for each reference coordinate p.
    levels=[(dense_sift(b),dense_sift(a))]
    kernel=cv2.getGaussianKernel(5,.67)
    for _ in range(3):
        prev=levels[-1]; shape=tuple(int(np.ceil(v/2)) for v in prev[0].shape[:2])
        levels.append(tuple(resize(cv2.sepFilter2D(im,-1,kernel,kernel,borderType=cv2.BORDER_REPLICATE),shape) for im in prev))
    flow=None
    for level in range(3,-1,-1):
        da,db=levels[level]; h,w=da.shape[:2]
        offsets=np.zeros((h,w,2)) if flow is None else resize(flow,(h,w))*2
        offsets=np.sign(offsets)*np.floor(np.abs(offsets)+.5)
        flow=discrete_flow(da,db,offsets[:,:,0],offsets[:,:,1],10 if level==3 else 2+level,
                           gamma=1.275*2**level,iterations=60 if level==3 else 30,hierarchy=2 if level==3 else 3-level)
    return flow


def sample_bicubic(image,x,y):
    xi=np.floor(x).astype(int); yi=np.floor(y).astype(int); out=np.zeros_like(image)
    for dy in (-1,0,1,2):
        wy=cubic(y-(yi+dy))
        for dx in (-1,0,1,2):
            weight=wy*cubic(x-(xi+dx))
            out+=image[np.clip(yi+dy,0,image.shape[0]-1),np.clip(xi+dx,0,image.shape[1]-1)]*weight[:,:,None]
    return out


def realign(a,b,mask):
    h,w=a.shape[:2]; flow=sift_flow(a,b)
    orth=np.array([1,0])
    if mask[:,-1].all(): orth=np.array([-1,0])
    if mask[0,:].all(): orth=np.array([0,1])
    if mask[-1,:].all(): orth=np.array([0,-1])
    yy,xx=np.mgrid[:h,:w]; corners=orth@np.array([[0,0,w-1,w-1],[0,h-1,0,h-1]])
    span=float(corners.max()-corners.min())
    if span==0: raise RuntimeError('Degenerate singleton LPAM patch')
    t=(orth[0]*xx+orth[1]*yy-corners.min())/span
    scale=1/(1+np.exp(-8*(t-.5)))
    x=np.clip(xx+flow[:,:,0]*scale,0,w-1); y=np.clip(yy+flow[:,:,1]*scale,0,h-1)
    warped=sample_bicubic(a,x,y)
    if not np.isfinite(warped).all(): raise RuntimeError('Non-finite localized SIFT Flow result')
    return warped,b


def improve(a,b,sa,sb):
    ma,mb=support(a),support(b); overlap=ma&mb
    seam=ndimage.binary_dilation(sa,structure=ndimage.generate_binary_structure(2,1))&ma&sb
    y,x=np.nonzero(seam.T); points=np.column_stack((x,y)) # MATLAB find column order
    info=dict(seam_pixels=len(points),patch_count=0,lpam_triggered=False,patch_boxes=[])
    if not len(points): return a,b,sa,sb,info
    errors,coords=quality(a,b,overlap,points)
    info.update(initial_seam_error_mean=float(errors.mean()),initial_seam_error_max=float(errors.max()))
    if errors.max()<=1.5*errors.mean(): return a,b,sa,sb,info
    threshold=hist_otsu(errors); marked=np.zeros(overlap.shape,np.uint8)
    for y0,y1,x0,x1 in coords[errors>=threshold]: marked[y0:y1,x0:x1]=1
    marked=cv2.morphologyEx(marked,cv2.MORPH_CLOSE,np.ones((10,10),np.uint8)).astype(bool)
    components,count=ndimage.label(marked,np.ones((3,3)))
    slices=ndimage.find_objects(components); outa=a.copy(); outb=b.copy(); outsa=sa.copy(); outsb=sb.copy()
    for box in slices:
        ys,xs=box
        wa,wb=realign(outa[box],outb[box],sa[box])
        la,lb,_=seam_cut(wa,wb,(sa[box],sb[box]))
        outa[box]=wa; outb[box]=wb; outsa[box]=la; outsb[box]=lb
        info['patch_boxes'].append([ys.start,ys.stop,xs.start,xs.stop])
    info.update(patch_count=count,lpam_triggered=count>0,otsu_threshold=threshold)
    return outa,outb,outsa,outsb,info


def composite(a,b,sa,sb):
    if np.any(sa&sb): raise RuntimeError('Seam labels overlap')
    result=a*sa[:,:,None]+b*sb[:,:,None]
    if not np.isfinite(result).all(): raise RuntimeError('Non-finite composite')
    return np.clip(np.floor(result*255+.5),0,255).astype(np.uint8)
