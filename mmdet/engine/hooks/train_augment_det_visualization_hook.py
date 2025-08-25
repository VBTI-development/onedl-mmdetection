# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmdet.models.utils import mask2ndarray
from mmdet.registry import HOOKS
from mmdet.structures.bbox import BaseBoxes


@HOOKS.register_module()
class TrainAugmentDetVisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize train augmentation.

    In the training phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. ``vis_backends`` takes effect if the user does not specify ``show``.
        You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(  # noqa: PLR0913
        self,
        draw: bool = False,
        interval: int = 50,
        show: bool = False,
        wait_time: float = 0.0,
        backend_args: Optional[dict] = None,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn(
                'The show is True, it means that only '
                'the prediction results are visualized '
                'without storing data, so vis_backends '
                'needs to be excluded.',
                stacklevel=1,
            )

        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Regularly check whether the loss is valid every n iterations.

        Args:
            runner (:obj:`Runner`): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict, Optional): Data from dataloader.
                Defaults to None.
            outputs (dict, Optional): Outputs from model. Defaults to None.
        """
        if self.draw is False or data_batch is None:
            return

        if self.every_n_train_iters(runner, self.interval):
            # There is no guarantee that the same batch of images
            # is visualized for each evaluation.
            total_curr_iter = runner.iter + batch_idx

            # Visualize only the first data
            data_sample = data_batch['data_samples'][0]
            img_path = data_sample.img_path
            img = data_batch['inputs'][0].permute(1, 2, 0).numpy()

            gt_instances = data_sample.gt_instances
            gt_bboxes = gt_instances.get('bboxes', None)
            if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
                gt_instances.bboxes = gt_bboxes.tensor
            gt_masks = gt_instances.get('masks', None)
            if gt_masks is not None:
                masks = mask2ndarray(gt_masks)
                gt_instances.masks = masks.astype(bool)
            data_sample.gt_instances = gt_instances

            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'train_img',
                img,
                data_sample=data_sample,
                show=self.show,
                draw_pred=False,
                wait_time=self.wait_time,
                step=total_curr_iter,
            )
