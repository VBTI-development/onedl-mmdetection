# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Any, Optional, Sequence

import numpy as np
from mmengine import MMLogger
from mmengine.evaluator import BaseMetric
from scipy.interpolate import splev, splprep
from scipy.optimize import linear_sum_assignment
from shapely import LineString, Polygon

from mmdet.registry import METRICS


def _continuous_cross_iou(xs, ys, img_height, img_width, width=30):
    """For each lane in xs, compute its Intersection Over Union (IoU) with each
    lane in ys using the area between each pair of points."""
    image = Polygon([(0, 0), (0, img_height - 1),
                     (img_width - 1, img_height - 1), (img_width - 1, 0)])
    xs = [
        LineString(lane).buffer(
            distance=width / 2., cap_style=1, join_style=2).intersection(image)
        for lane in xs
    ]
    ys = [
        LineString(lane).buffer(
            distance=width / 2., cap_style=1, join_style=2).intersection(image)
        for lane in ys
    ]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def culane_metric(
    annotation,
    prediction,
    width=30,
    iou_thresholds=[0.5],
):
    ann_lines = annotation['lines']
    pred_lines = prediction['lines']
    image_width = annotation['org_width']
    image_height = annotation['org_height']

    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(ann_lines) != 0 else len(pred_lines)
        fn = 0 if len(pred_lines) != 0 else len(ann_lines)
        _metric[thr] = [tp, fp, fn]

    interp_pred = np.array([interp(line, n=50) for line in pred_lines],
                           dtype=object)  # (4, 50, 2)
    anno = np.array([np.array(line) for line in ann_lines], dtype=object)

    ious = _continuous_cross_iou(
        interp_pred,
        anno,
        width=width,
        img_height=image_height,
        img_width=image_width)

    row_ind, col_ind = linear_sum_assignment(1 - ious)

    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred_lines) - tp
        fn = len(ann_lines) - tp
        _metric[thr] = [tp, fp, fn]
    return _metric


def eval_predictions(results,
                     iou_thresholds=[0.5],
                     width=30,
                     sequential=False,
                     logger=None):
    annotations, predictions = zip(*results)
    if sequential:
        results = map(
            partial(
                culane_metric,
                width=width,
                iou_thresholds=iou_thresholds,
            ),
            annotations,
            predictions,
        )
    else:
        from itertools import repeat
        from multiprocessing import Pool, cpu_count
        with Pool(cpu_count()) as p:
            results = p.starmap(
                culane_metric,
                zip(
                    annotations,
                    predictions,
                    repeat(width),
                    repeat(iou_thresholds),
                ))

    results = list(results)
    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = \
        0, 0, 0, 0, 0, 0
    ret = {}
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp != 0 else 0
        logger.info('iou thr: {:.2f}, tp: {}, fp: {}, fn: {},'
                    'precision: {}, recall: {}, f1: {}'.format(
                        thr, tp, fp, fn, precision, recall, f1))
        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ret[thr] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    if len(iou_thresholds) > 2:
        logger.info('mean result, total_tp: {}, total_fp: {}, total_fn: {},'
                    'precision: {}, recall: {}, f1: {}'.format(
                        total_tp, total_fp, total_fn, mean_prec, mean_recall,
                        mean_f1))
        ret['mean'] = {
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': mean_prec,
            'Recall': mean_recall,
            'F1': mean_f1
        }
    flat_metrics = {}

    if 'mean' in ret:
        for k, v in ret['mean'].items():
            flat_metrics[k] = v

    for k in set(ret.keys()) - {'mean'}:
        for kk, vv in ret[k].items():
            flat_metrics[f'{kk}_{int(k * 100)}'] = vv

    return flat_metrics


@METRICS.register_module()
class CULaneMetric(BaseMetric):
    default_prefix = 'culane'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None,
                 iou_thresholds: Optional[Sequence[float]] = (0.5, )):
        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
            collect_dir=collect_dir)
        self.iou_thresholds = iou_thresholds

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            prediction = dict()
            pred_instances = data_sample['pred_instances']
            prediction['lines'] = pred_instances['lines']

            ground_truth = dict()
            ground_truth['lines'] = data_sample['org_lines']
            ground_truth['org_height'] = data_sample['org_height']
            ground_truth['org_width'] = data_sample['org_width']
            self.results.append((ground_truth, prediction))

    def compute_metrics(self, results) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        metrics = eval_predictions(
            results, self.iou_thresholds, logger=logger, sequential=True)
        return metrics
