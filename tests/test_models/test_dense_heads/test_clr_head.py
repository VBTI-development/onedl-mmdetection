from unittest import TestCase

import numpy as np
import torch

from mmdet import *  # noqa
from mmdet.models.dense_heads import CLRHead
from mmdet.models.dense_heads.clr_head.losses.lineiou_loss import liou_loss


# Test case 1: Simple non-overlapping lines
def create_line(x1, y1, start_pos, length, offsets, valid=1):
    """Create a line with 78 features: [ valid, x1, y1, start_pos, ?, length,
    ...72_offsets]"""
    line = np.zeros(78)
    line[0] = 0
    line[1] = float(valid)
    line[2] = x1  # x coordinate
    line[3] = y1  # y coordinate
    line[4] = start_pos  # starting position
    line[5] = 0  # placeholder (from your CUDA code)
    line[6] = length  # length
    line[7:] = np.array(offsets[:71])  # 72 offset values
    return line


def test_liou_loss():

    # Line 1: Horizontal line at top
    offsets1 = [10.0] * 72
    line1 = create_line(100, 50, 0.1, 30, offsets1)

    # Line 2: Vertical line (non-overlapping)
    offsets2 = list(range(72))
    line2 = create_line(200, 100, 0.2, 40, offsets2)

    line1 = torch.tensor(line1)
    line2 = torch.tensor(line2)

    assert torch.gt(liou_loss(line1, line2, 100), 0.9)
    assert torch.allclose(
        liou_loss(line1, line1, 100), torch.tensor(0.0, dtype=torch.float64))

    # Line 2: Similar horizontal line
    offsets3 = [12.0] * 72
    line3 = create_line(110, 52, 0.12, 28, offsets3)
    line3 = torch.tensor(line3)

    assert torch.lt(liou_loss(line1, line3, 100), 0.12)


class TestCLRHead(TestCase):

    def test_init(self):
        """Test init clr head."""
        clr_head = CLRHead(num_classes=1)
        self.assertTrue(clr_head.reg_modules)
        self.assertTrue(clr_head.cls_modules)

    def test_clr_head_forward(self):
        """Tests clr head can do forward and predict."""
        s = 256
        clr_head = CLRHead(num_classes=1, img_w=s, img_h=s)

        # Anchor head expects a multiple levels of features per image
        feats = list(
            torch.rand(
                1,
                64,
                s // (2**(i + 2)),
                s // (2**(i + 2)),
                dtype=torch.float32) for i in range(clr_head.refine_layers))
        batch_data_samples = []
        preds = clr_head.forward(feats, batch_data_samples)

        assert preds.sum() != 0
