# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Tuple, Union

import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from mmcv import BaseTransform, to_tensor
from mmengine.registry import TRANSFORMS
from mmengine.structures import BaseDataElement, PixelData
from scipy.interpolate import InterpolatedUnivariateSpline


@TRANSFORMS.register_module()
class ImageAug:

    def __init__(
        self,
        pipeline: dict,
    ):
        self.img_w, self.img_h = self._get_resize_width_height(pipeline)

        img_transforms = []
        for aug in pipeline:
            p = aug['p']
            if aug['name'] != 'OneOf':
                img_transforms.append(
                    iaa.Sometimes(
                        p=p,
                        then_list=getattr(iaa,
                                          aug['name'])(**aug['parameters'])))
            else:
                img_transforms.append(
                    iaa.Sometimes(
                        p=p,
                        then_list=iaa.OneOf([
                            getattr(iaa, aug_['name'])(**aug_['parameters'])
                            for aug_ in aug['transforms']
                        ])))
        self.pipeline = iaa.Sequential(img_transforms)

    def _get_resize_width_height(self, transforms):
        for transform in transforms:
            if transform['name'] == 'Resize':
                size = transform['parameters']['size']
                return size['width'], size['height']
        msg = 'Resize transform not found'
        raise ValueError(msg)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))
        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)
        return lanes

    def transform(self, result):
        img_org = result['img']

        # record original image size
        result['org_width'], result['org_height'] = img_org.shape[
            1], img_org.shape[0]

        if 'instances' in result:
            lines = [instance['line'] for instance in result['instances']]
            result['org_lines'] = lines
            line_strings_org = self.lane_to_linestrings(lines)
            line_strings_org = LineStringsOnImage(
                line_strings_org, shape=img_org.shape)
            if 'gt_seg_map' in result:
                mask_org = SegmentationMapsOnImage(
                    result['gt_seg_map'], shape=img_org.shape)
                img, line_strings, seg = self.pipeline(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org,
                    segmentation_maps=mask_org)
                result['gt_seg_map'] = seg.get_arr()
            else:
                img, line_strings = self.pipeline(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org)
            line_strings.clip_out_of_image_()
            result['lines'] = self.linestrings_to_lanes(line_strings)
        else:
            img = self.pipeline(image=img_org.copy().astype(np.uint8))
        result['img'] = img
        return result

    def __call__(self, result):
        return self.transform(result)


@TRANSFORMS.register_module()
class LinesToArray(BaseTransform):

    def __init__(
        self,
        num_points: int = 72,
        max_lines: int = 4,
        img_height: int = 320,
        img_width: int = 800,
    ):
        self.num_points = num_points
        self.max_lines = max_lines
        self.img_h = img_height
        self.img_w = img_width

        self.n_offsets = self.num_points
        self.n_strips = self.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.sample_y = range(589, 230, -20)

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(
            y[::-1], x[::-1], k=min(3,
                                    len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y)
                                            & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line
        # using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(
            two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform(self, result):
        img_w, img_h = self.img_w, self.img_h

        if 'lines' not in result:
            return result

        old_lanes = result['lines']

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[
            x * self.img_w / float(img_w), y * self.img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]
        # create transformed annotations
        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, S+1 coordinates
        lines_array = np.ones((self.max_lines, 2 + 1 + 1 + 2 + self.n_offsets),
                              dtype=np.float32) * -1e5
        line_endpoints = np.ones((self.max_lines, 2))
        # lanes are invalid by default
        lines_array[:, 0] = 1
        lines_array[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lines:
                break

            try:
                xs_outside_image, xs_inside_image = self.sample_lane(
                    lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lines_array[lane_idx, 0] = 0
            lines_array[lane_idx, 1] = 1
            lines_array[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lines_array[lane_idx, 3] = xs_inside_image[0]

            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = math.atan(
                    i * self.strip_size /
                    (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)

            theta_far = sum(thetas) / len(thetas)

            # lanes[lane_idx,
            #       4] = (theta_closest + theta_far) / 2  # averaged angle
            lines_array[lane_idx, 4] = theta_far
            lines_array[lane_idx, 5] = len(xs_inside_image)
            lines_array[lane_idx, 6:6 + len(all_xs)] = all_xs
            line_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            line_endpoints[lane_idx, 1] = xs_inside_image[-1]

        result['lines_array'] = lines_array
        result['line_endpoints'] = line_endpoints
        return result


@TRANSFORMS.register_module()
class PackLineDetectionInputs(BaseTransform):

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        data_sample = BaseDataElement()
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = img

        if 'lines_array' in results:
            data_sample.lines_array = results['lines_array']
            data_sample.line_endpoints = results['line_endpoints']
            data_sample.lines = results['lines']
            data_sample.org_lines = results['org_lines']
        data_sample.org_height = results['org_height']
        data_sample.org_width = results['org_width']

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        packed_results['data_samples'] = data_sample
        return packed_results
