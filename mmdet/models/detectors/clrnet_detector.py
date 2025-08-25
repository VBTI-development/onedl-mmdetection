# Copyright (c) VBTI. All rights reserved.
from mmengine import MODELS

from .single_stage import SingleStageDetector


@MODELS.register_module()
class CLRNet(SingleStageDetector):

    def __init__(self, backbone, neck=None, head=None, data_preprocessor=None):
        super(CLRNet, self).__init__(
            backbone, neck, head, data_preprocessor=data_preprocessor)
        self.aggregator = None

    def _forward(self, inputs, batch, *args, **kwargs):
        x = self.extract_feat(inputs['img'])
        results = self.bbox_head.forward(x, batch)
        return results
