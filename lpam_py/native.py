import ctypes as ct
from pathlib import Path
import numpy as np

BUILD=Path(__file__).resolve().parent/'_native'
_libs={}
DOUBLE=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
BYTE=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
INT=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')


def function(name,symbol,types):
    if name not in _libs: _libs[name]=ct.CDLL(str(BUILD/f'{name}.dll'))
    fn=getattr(_libs[name],symbol); fn.argtypes=types; fn.restype=ct.c_int
    return fn


def check(code):
    if code==2: raise MemoryError('Original native kernel allocation failed')
    if code: raise RuntimeError(f'Original native kernel error {code}')


def dense_sift(rgb):
    rgb=np.ascontiguousarray(rgb,dtype=np.float64); h,w=rgb.shape[:2]
    out=np.zeros((h,w,128),np.uint8)
    check(function('dense','dense_sift',[DOUBLE,ct.c_int,ct.c_int,BYTE])(rgb,h,w,out))
    return out


def discrete_flow(a,b,ox,oy,window,alpha=510.,d=10200.,gamma=1.275,iterations=30,hierarchy=0):
    h,w=a.shape[:2]; a=np.ascontiguousarray(a,np.uint8); b=np.ascontiguousarray(b,np.uint8)
    ox=np.ascontiguousarray(ox,np.int32); oy=np.ascontiguousarray(oy,np.int32)
    wx=np.full((h,w),window,np.int32); wy=wx.copy(); out=np.zeros((h,w,2),np.float64)
    fn=function('flow','discrete_flow',[BYTE,BYTE,ct.c_int,ct.c_int,INT,INT,INT,INT,
        ct.c_double,ct.c_double,ct.c_double,ct.c_int,ct.c_int,ct.c_int,DOUBLE])
    check(fn(a,b,h,w,ox,oy,wx,wy,alpha,d,gamma,iterations,hierarchy,window,out))
    return out


def graph_cut(terminal,u,v,cost):
    terminal=np.ascontiguousarray(terminal,np.float64); u=np.ascontiguousarray(u,np.int32)
    v=np.ascontiguousarray(v,np.int32); cost=np.ascontiguousarray(cost,np.float64)
    labels=np.zeros(len(terminal),np.uint8); energy=np.zeros(1,np.float64)
    fn=function('cut','graph_cut',[ct.c_int,ct.c_int,DOUBLE,INT,INT,DOUBLE,BYTE,DOUBLE])
    check(fn(len(labels),len(cost),terminal,u,v,cost,labels,energy))
    return labels,float(energy[0])
