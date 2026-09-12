import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from baseline import mask_metrics,calibrate_threshold,score_image

class BaselineTests(unittest.TestCase):
    def test_false_positive_and_miss_are_counted(self):
        m=mask_metrics([[1,1],[0,0]],[[1,0],[1,0]])
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']),(1,1,1,1))
        self.assertAlmostEqual(m['dice'],.5)
        self.assertAlmostEqual(m['iou'],1/3)

    def test_empty_masks_have_documented_behavior(self):
        m=mask_metrics(np.zeros((3,3)),np.zeros((3,3)))
        self.assertEqual(m['dice'],1.)
        self.assertIsNone(m['precision'])
        self.assertIsNone(m['recall'])

    def test_normal_cutoff_respects_tail(self):
        scores=np.arange(100,dtype=float).reshape(10,10)/100
        cutoff=calibrate_threshold([scores],.1)
        self.assertLessEqual(float(np.mean(scores>cutoff)),.1)

    def test_residual_rejects_accidental_broadcasting(self):
        with self.assertRaises(ValueError):score_image(np.zeros((2,3)),np.zeros((3,)))

    def test_invalid_calibration_cannot_succeed(self):
        with self.assertRaises(ValueError):calibrate_threshold([])
        with self.assertRaises(ValueError):calibrate_threshold([np.zeros((2,2))],1)

if __name__=='__main__':unittest.main()
