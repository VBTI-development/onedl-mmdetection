# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

from mmcv.transforms import BaseTransform
from onedl.core import read

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadLineAnnotations(BaseTransform):

    def __init__(
        self,
        with_segmentation: bool = False,
    ):
        super().__init__()
        self.with_segmentation = with_segmentation

    def __call__(self,
                 results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        return self.transform(results)

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function. All subclass of BaseTransform should
        override this method.

        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if self.with_segmentation:
            self._load_segmentation(results)
        self._load_lines(results)
        return results

    def _load_lines(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        results['lines'] = [
            instance['line'] for instance in results['instances']
        ]

    def _load_segmentation(self, results):
        seg_map_array = read(results['segmentation_path']).to_numpy(False)
        if len(seg_map_array.shape) > 2:
            seg_map_array = seg_map_array[:, :, 0]
        seg_map_array = seg_map_array.squeeze()
        results.update({'gt_seg_map': seg_map_array})


@TRANSFORMS.register_module()
class LoadOneDLSegmentation(BaseTransform):

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        results['gt_seg_map'] = read(
            results['segmentation_path']).to_numpy(False)
        return results
