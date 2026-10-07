import unittest
import numpy as np
from lpam_py.native import dense_sift,graph_cut
from lpam_py.core import sift_flow,seam_cut,improve,composite,quality,sample_bicubic
from lpam_py.geometry import dlt,estimate


class CoreTests(unittest.TestCase):
    def test_cut_labels(self):
        labels,energy=graph_cut(np.array([[100.,0],[0,100.]]),np.array([0]),np.array([1]),np.ones(1))
        np.testing.assert_array_equal(labels,[0,1]); self.assertEqual(energy,1.)

    def test_sift_flow_identity_and_translation(self):
        rng=np.random.RandomState(2); a=rng.rand(48,64,3); b=np.roll(a,3,axis=1)
        self.assertEqual(dense_sift(a).shape,(48,64,128))
        self.assertEqual(float(abs(sift_flow(a,a)).max()),0.)
        flow=sift_flow(a,b)
        self.assertLess(abs(np.median(flow[12:-12,12:-12,0])+3),.5)
        self.assertLess(abs(np.median(flow[12:-12,12:-12,1])),.5)

    def test_local_boundary_labels(self):
        a=np.full((32,40,3),.2); b=np.full_like(a,.3)
        sa=np.zeros((32,40),bool); sa[:,:20]=True; sb=~sa
        la,lb,_=seam_cut(a,b,(sa,sb))
        edge=np.zeros_like(sa); edge[[0,-1],:]=True; edge[:,[0,-1]]=True
        np.testing.assert_array_equal(la[edge],sa[edge]); self.assertTrue((la^lb).all())

    def test_quality_and_unchanged_safe_region(self):
        rng=np.random.RandomState(5); a=rng.rand(48,64,3)
        sa=np.zeros((48,64),bool); sa[:,:32]=True; sb=~sa
        errors,_=quality(a,a,np.ones((48,64),bool),np.array([[24,32]]))
        self.assertLess(abs(errors[0]),1e-10)
        wa,wb,la,lb,info=improve(a,a,sa,sb)
        self.assertFalse(info['lpam_triggered'])
        np.testing.assert_array_equal(composite(wa,wb,la,lb),composite(a,a,sa,sb))

    def test_final_uses_realigned_image(self):
        a=np.zeros((32,40,3)); b=a.copy(); wa=a.copy(); wa[:,:20,0]=1
        sa=np.zeros((32,40),bool); sa[:,:20]=True; sb=~sa
        result=composite(wa,b,sa,sb)
        self.assertEqual(result[:,:20,0].min(),255)
        yy,xx=np.mgrid[:32,:40]
        np.testing.assert_allclose(sample_bicubic(wa,xx.astype(float),yy.astype(float)),wa)

    def test_normalized_homography(self):
        rng=np.random.RandomState(1); a=rng.rand(40,2)*100+1; b=a+[8,-5]
        h,n=estimate(a,b); predicted=np.column_stack((a,np.ones(40)))@h.T
        np.testing.assert_allclose(predicted[:,:2]/predicted[:,2:],b,atol=1e-7)
        self.assertEqual(n,40)


if __name__=='__main__': unittest.main()
