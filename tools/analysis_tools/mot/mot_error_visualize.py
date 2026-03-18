# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import mmcv
import numpy as np
import pandas as pd
from mmengine import Config
from mmengine.logging import print_log
from mmengine.registry import init_default_scope
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset

from mmdet.registry import DATASETS
from mmdet.utils import imshow_mot_errors

MOT_COLUMNS = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence']
LARGE_COST = 1e6


def parse_args():
    parser = argparse.ArgumentParser(
        description='visualize errors for multiple object tracking')
    parser.add_argument('config', help='path of the config file')
    parser.add_argument(
        '--result-dir',
        required=True,
        help='directory of the inference result')
    parser.add_argument(
        '--out-dir',
        '--output-dir',
        dest='out_dir',
        help='directory where painted images or videos will be saved')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to show the results on the fly')
    parser.add_argument(
        '--fps', type=int, default=3, help='FPS of the output video')
    parser.add_argument(
        '--backend',
        type=str,
        choices=['cv2', 'plt'],
        default='cv2',
        help='backend of visualization')
    args = parser.parse_args()
    return args


def read_mot_txt(file_path: str) -> pd.DataFrame:
    """Read a MOT challenge txt file.

    Returns:
        DataFrame: Columns are FrameId, Id, X, Y, Width, Height, Confidence.
    """
    try:
        data = pd.read_csv(
            file_path, header=None, sep=r',|\s+', engine='python')
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=MOT_COLUMNS)

    data = data.dropna(axis=1, how='all')
    if data.shape[1] < 6:
        raise ValueError(f'Invalid MOT file format: {file_path}')

    if data.shape[1] == 6:
        data[6] = 1.0

    data = data.iloc[:, :7].copy()
    data.columns = MOT_COLUMNS
    data['FrameId'] = data['FrameId'].astype(int)
    data['Id'] = data['Id'].astype(int)
    for col in ['X', 'Y', 'Width', 'Height', 'Confidence']:
        data[col] = data[col].astype(float)
    return data


def remap_frame_ids(data: pd.DataFrame) -> pd.DataFrame:
    """Remap frame ids in a MOT txt table to be continuous from 1."""
    if data.empty:
        return data

    frame_ids = sorted(data['FrameId'].unique())
    id_map = {fid: idx + 1 for idx, fid in enumerate(frame_ids)}
    data = data.copy()
    data['FrameId'] = data['FrameId'].map(id_map).astype(int)
    return data


def xywh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Convert bboxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    if bboxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    xyxy = bboxes.astype(np.float32, copy=True)
    xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
    xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
    return xyxy


def bbox_iou_matrix(gt_bboxes: np.ndarray,
                    pred_bboxes: np.ndarray) -> np.ndarray:
    """Compute pair-wise IoU matrix for two bbox sets in xyxy format."""
    if gt_bboxes.shape[0] == 0 or pred_bboxes.shape[0] == 0:
        return np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]),
                        dtype=np.float32)

    lt = np.maximum(gt_bboxes[:, None, :2], pred_bboxes[None, :, :2])
    rb = np.minimum(gt_bboxes[:, None, 2:], pred_bboxes[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    overlap = wh[..., 0] * wh[..., 1]

    gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
        gt_bboxes[:, 3] - gt_bboxes[:, 1])
    pred_areas = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (
        pred_bboxes[:, 3] - pred_bboxes[:, 1])
    union = gt_areas[:, None] + pred_areas[None, :] - overlap
    union = np.maximum(union, 1e-6)

    return overlap / union


def get_bbox_row(frame_data: pd.DataFrame,
                 instance_id: int) -> Optional[np.ndarray]:
    """Get the first bbox row for one id in one frame."""
    matched = frame_data[frame_data['Id'] == instance_id]
    if matched.empty:
        return None

    row = matched.iloc[0]
    return np.array([
        row['X'], row['Y'], row['X'] + row['Width'], row['Y'] + row['Height'],
        row['Confidence']
    ],
                    dtype=np.float32)


def compare_res_gts(
    results_dir: str,
    dataset: Dataset,
    video_name: str,
    iou_thr: float = 0.5
) -> Tuple[Dict[int, Dict[str, List[int]]], pd.DataFrame, pd.DataFrame]:
    """Compare prediction result with GT and return frame-level MOT errors.

    Args:
        results_dir (str): the directory of the MOT results.
        dataset (Dataset): MOT dataset of the video to be evaluated.
        video_name (str): Name of the video to be evaluated.
        iou_thr (float): IoU threshold used for matching.

    Returns:
        tuple: (events_by_frame, res, gt), where events_by_frame stores
        lists of FP, FN and IDSW ids in each frame.
    """
    if 'half-train' in dataset.ann_file:
        gt_file = osp.join(dataset.data_prefix['img_path'],
                           f'{video_name}/gt/gt_half-train.txt')
    elif 'half-val' in dataset.ann_file:
        gt_file = osp.join(dataset.data_prefix['img_path'],
                           f'{video_name}/gt/gt_half-val.txt')
    else:
        gt_file = osp.join(dataset.data_prefix['img_path'],
                           f'{video_name}/gt/gt.txt')

    gt = remap_frame_ids(read_mot_txt(gt_file))
    res_file = osp.join(results_dir, f'{video_name}.txt')
    res = read_mot_txt(res_file)

    events_by_frame: Dict[int, Dict[str, List[int]]] = defaultdict(
        lambda: dict(fp=[], fn=[], idsw=[]))
    last_matches: Dict[int, int] = {}

    frame_ids = sorted(
        set(gt['FrameId'].tolist()) | set(res['FrameId'].tolist()))
    for frame_id in frame_ids:
        frame_gt = gt[gt['FrameId'] == frame_id]
        frame_res = res[res['FrameId'] == frame_id]

        gt_ids = frame_gt['Id'].to_numpy(dtype=np.int64)
        pred_ids = frame_res['Id'].to_numpy(dtype=np.int64)
        gt_boxes = xywh_to_xyxy(frame_gt[['X', 'Y', 'Width', 'Height'
                                          ]].to_numpy(dtype=np.float32))
        pred_boxes = xywh_to_xyxy(frame_res[['X', 'Y', 'Width', 'Height'
                                             ]].to_numpy(dtype=np.float32))

        matched_gt_inds = set()
        matched_pred_inds = set()
        switched_ids: List[int] = []

        if gt_boxes.shape[0] > 0 and pred_boxes.shape[0] > 0:
            ious = bbox_iou_matrix(gt_boxes, pred_boxes)
            costs = 1 - ious
            costs[ious < iou_thr] = LARGE_COST
            row_inds, col_inds = linear_sum_assignment(costs)

            valid = costs[row_inds, col_inds] < LARGE_COST
            row_inds = row_inds[valid]
            col_inds = col_inds[valid]

            matched_gt_inds = set(row_inds.tolist())
            matched_pred_inds = set(col_inds.tolist())

            for gt_ind, pred_ind in zip(row_inds, col_inds):
                gt_id = int(gt_ids[gt_ind])
                pred_id = int(pred_ids[pred_ind])
                prev_id = last_matches.get(gt_id)
                if prev_id is not None and prev_id != pred_id:
                    switched_ids.append(pred_id)
                last_matches[gt_id] = pred_id

        fp_ids = [
            int(pred_ids[i]) for i in range(len(pred_ids))
            if i not in matched_pred_inds
        ]
        fn_ids = [
            int(gt_ids[i]) for i in range(len(gt_ids))
            if i not in matched_gt_inds
        ]

        if len(fp_ids) > 0 or len(fn_ids) > 0 or len(switched_ids) > 0:
            events_by_frame[frame_id] = dict(
                fp=fp_ids, fn=fn_ids, idsw=switched_ids)

    return dict(events_by_frame), res, gt


def main():
    args = parse_args()

    assert args.show or args.out_dir, \
        ('Please specify at least one operation (show the results '
         '/ save the results) with the argument "--show" or "--out-dir"')
    assert args.result_dir is not None, '--result-dir is required.'

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    print_log('This script visualizes the error for multiple object tracking. '
              'By Default, the red bounding box denotes false positive, '
              'the yellow bounding box denotes the false negative '
              'and the blue bounding box denotes ID switch.')

    cfg = Config.fromfile(args.config)

    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataset = DATASETS.build(cfg.val_dataloader.dataset)

    # create index from frame_id to filename
    filenames_dict = dict()
    for i in range(len(dataset)):
        video_info = dataset.get_data_info(i)
        # the `data_info['file_name']` usually has the same format
        # with "MOT17-09-DPM/img1/000003.jpg"
        # split with both '\' and '/' to be compatible with different OS.
        for data_info in video_info['images']:
            split_path = re.split(r'[\\/]', data_info['file_name'])
            video_name = split_path[-3]
            frame_id = int(data_info['frame_id'] + 1)
            if video_name not in filenames_dict:
                filenames_dict[video_name] = dict()
        # the data_info['img_path'] usually has the same format
        # with `img_path_prefix + "MOT17-09-DPM/img1/000003.jpg"`
            filenames_dict[video_name][frame_id] = data_info['img_path']
    video_names = tuple(filenames_dict.keys())

    for video_name in video_names:
        print_log(f'Start processing video {video_name}')

        events_by_frame, res, gt = compare_res_gts(args.result_dir, dataset,
                                                   video_name)

        frames_id_list = sorted(events_by_frame.keys())
        if len(frames_id_list) == 0:
            print_log(f'No tracking error is found in video {video_name}')
            continue

        if args.out_dir is not None:
            os.makedirs(osp.join(args.out_dir, video_name), exist_ok=True)

        for frame_id in frames_id_list:
            # events in the current frame
            cur_events = events_by_frame[frame_id]
            cur_res = res[res['FrameId'] == frame_id]
            cur_gt = gt[gt['FrameId'] == frame_id]
            # path of image
            if frame_id not in filenames_dict[video_name]:
                continue
            img = filenames_dict[video_name][frame_id]

            bboxes, ids, error_types = [], [], []
            for hid in cur_events['fp']:
                bbox = get_bbox_row(cur_res, hid)
                if bbox is None:
                    continue
                bboxes.append(bbox)
                ids.append(hid)
                # error_type = 0 denotes false positive error
                error_types.append(0)
            for oid in cur_events['fn']:
                bbox = get_bbox_row(cur_gt, oid)
                if bbox is None:
                    continue
                bboxes.append(bbox)
                ids.append(-1)
                # error_type = 1 denotes false negative error
                error_types.append(1)
            for hid in cur_events['idsw']:
                bbox = get_bbox_row(cur_res, hid)
                if bbox is None:
                    continue
                bboxes.append(bbox)
                ids.append(hid)
                # error_type = 2 denotes id switch
                error_types.append(2)
            if len(bboxes) == 0:
                bboxes = np.zeros((0, 5), dtype=np.float32)
            else:
                bboxes = np.asarray(bboxes, dtype=np.float32)
            ids = np.asarray(ids, dtype=np.int32)
            error_types = np.asarray(error_types, dtype=np.int32)
            imshow_mot_errors(
                img,
                bboxes,
                ids,
                error_types,
                show=args.show,
                out_file=osp.join(args.out_dir,
                                  f'{video_name}/{frame_id:06d}.jpg')
                if args.out_dir else None,
                backend=args.backend)

        if args.out_dir is not None:
            print_log(f'Done! Visualization images are saved in '
                      f'\'{args.out_dir}/{video_name}\'')
            mmcv.frames2video(
                f'{args.out_dir}/{video_name}',
                f'{args.out_dir}/{video_name}.mp4',
                fps=args.fps,
                fourcc='mp4v',
                start=frames_id_list[0],
                end=frames_id_list[-1],
                show_progress=False)
            print_log(
                f'Done! Visualization video is saved as '
                f'\'{args.out_dir}/{video_name}.mp4\' with a FPS of {args.fps}'
            )


if __name__ == '__main__':
    main()
