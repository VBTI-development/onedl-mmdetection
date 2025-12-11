# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.testing import demo_track_inputs, random_boxes
from mmdet.utils import register_all_modules


@pytest.fixture()
def tracker_setup():
    register_all_modules(init_default_scope=True)
    cfg = dict(
        type='ByteTracker',
        motion=dict(type='KalmanFilter'),
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_tentatives=3,
        num_frames_retain=30)
    tracker = MODELS.build(cfg)
    tracker.kf = TASK_UTILS.build(dict(type='KalmanFilter'))
    return {
        'tracker': tracker,
        'num_frames_retain': cfg['num_frames_retain'],
        'num_objs': 30
    }


@pytest.mark.parametrize('with_mask', [True, False])
def test_init(tracker_setup, with_mask):
    tracker = tracker_setup['tracker']
    num_objs = tracker_setup['num_objs']

    bboxes = random_boxes(num_objs, 512)
    labels = torch.zeros(num_objs)
    scores = torch.ones(num_objs)
    ids = torch.arange(num_objs)

    if with_mask:
        masks = torch.randint(0, 2, (num_objs, 28, 28)).bool()
        tracker.update(
            ids=ids,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            masks=masks,
            frame_ids=0)
    else:
        tracker.update(
            ids=ids, bboxes=bboxes, scores=scores, labels=labels, frame_ids=0)

    assert tracker.ids == list(ids)

    memo_items = ['ids', 'bboxes', 'scores', 'labels', 'frame_ids']
    if with_mask:
        memo_items = [
            'ids', 'bboxes', 'scores', 'labels', 'masks', 'frame_ids'
        ]
    assert tracker.memo_items == memo_items


@pytest.mark.parametrize('with_mask', [True, False])
def test_track(tracker_setup, with_mask):
    tracker = tracker_setup['tracker']

    with torch.no_grad():
        packed_inputs = demo_track_inputs(
            batch_size=1, num_frames=2, with_mask=with_mask)
        track_data_sample = packed_inputs['data_samples'][0]
        video_len = len(track_data_sample)
        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            img_data_sample.pred_instances = \
                img_data_sample.gt_instances.clone()
            if with_mask:
                img_data_sample.pred_instances.masks = \
                    img_data_sample.pred_instances.masks.to_tensor(float, 'cpu')  # noqa: E501
            # add fake scores
            scores = torch.ones(len(img_data_sample.gt_instances.bboxes))
            img_data_sample.pred_instances.scores = torch.FloatTensor(scores)

            pred_track_instances = tracker.track(data_sample=img_data_sample)

            bboxes = pred_track_instances.bboxes
            labels = pred_track_instances.labels

            assert bboxes.shape[1] == 4
            assert bboxes.shape[0] == labels.shape[0]

            if with_mask:
                masks = pred_track_instances.masks
                assert len(masks) == labels.shape[0]
