"""Verified resume and owned-worker timeout cleanup, without benchmark writes."""
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from lpam_py.run import process,pair_key,digest


class RunnerTests(unittest.TestCase):
    def configuration(self,root):
        repo=Path(__file__).resolve().parents[1]
        pair=dict(scene='unit',pair_id='12',pair_name='unit_p12',gt_path='',
                  left_source=str(repo/'Imgs/1_l.jpg'),right_source=str(repo/'Imgs/1_r.jpg'),
                  final_pair_dir=str(root/'unit'))
        args=Namespace(benchmark='stitchbench',result_root=root,max_input_edge=2048,max_out_height=8000,
                       max_canvas_pixels=16000000,lpips_max_side=1024,timeout=1,skip_existing=True,force=False)
        return pair,args

    def test_verified_resume_does_not_reinfer(self):
        with tempfile.TemporaryDirectory(prefix='lpam-test-') as tmp:
            pair,args=self.configuration(Path(tmp))
            output=Path(pair['final_pair_dir'])/'lpam/stitch_result.png'; output.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(output),np.zeros((32,40,3),np.uint8)))
            key,_=pair_key(pair,args,'unit-version'); timestamp=output.stat().st_mtime_ns
            previous=dict(cache_key=key,output_sha256=digest(output),inference_status='success',evaluation_status='success')
            self.assertIs(process(pair,args,'unit-version',previous,None,None,'cpu'),previous)
            self.assertEqual(output.stat().st_mtime_ns,timestamp)

    def test_timeout_is_failure_and_removes_empty_method_dir(self):
        with tempfile.TemporaryDirectory(prefix='lpam-test-') as tmp:
            pair,args=self.configuration(Path(tmp)); args.skip_existing=False
            record=process(pair,args,'unit-version',{},None,None,'cpu')
            self.assertEqual(record['inference_status'],'failed')
            self.assertEqual(record['evaluation_status'],'not_attempted')
            self.assertEqual(record['failure_stage'],'inference_timeout')
            self.assertIn('TimeoutExpired',record['failure_reason'])
            self.assertFalse((Path(pair['final_pair_dir'])/'lpam').exists())


if __name__=='__main__': unittest.main()
