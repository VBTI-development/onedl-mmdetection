# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from pathlib import Path

from mmdet.datasets import DSDLDetDataset

try:
    from dsdl.dataset import DSDLDataset
except ImportError:
    DSDLDataset = None


class TestDSDLDetDataset(unittest.TestCase):

    def test_dsdldet_init(self):
        if DSDLDataset is not None:
            dataset = DSDLDetDataset(
                data_root=Path(__file__).parent.parent / 'data/dsdl_det',
                ann_file='set-train/train.yaml')
            dataset.full_init()

            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(dataset[0]['instances']), 4)
            self.assertEqual(dataset.get_cat_ids(0), [3, 0, 0, 1])
        else:
            ImportWarning('Package `dsdl` is not installed.')
