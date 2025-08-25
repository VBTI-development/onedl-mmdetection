# Copyright (c) VBTI. All rights reserved.
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from .lane import Lane
from .losses.lineiou_loss import liou_loss
from .utils.dynamic_assign import assign
from .utils.roi_gather import LinearModule, ROIGather
from .utils.seg_decoder import SegDecoder

try:
    from mmcv.ops import line_nms as nms
except ImportError:
    logger: MMLogger = MMLogger.get_current_instance()
    logger.warning(
        'NMS is not compiled. Most likely when installing the package cuda '
        'was not available.\n'
        'This model cannot work without CUDA, therefore you need to recompile'
        ' the package with CUDA support.\n'
        'To make sure that the package is compiled with CUDA support use '
        'FORCE_CUDA=1 mim install -e .\n')
    nms = None


def visualize_lane_priors_tensor(prior_tensors, num_points=72):
    """Visualize lane priors on an image using PyTorch tensors.

    Args:
    - image_tensor: torch.Tensor of shape (3, H, W) representing the input
        image
    - prior_tensors: torch.Tensor of shape (num_priors, num_components) where
        - num_components = 4 + num_offsets
        - prior_tensors[:, 1]: foreground/background probability
        - prior_tensors[:, 2]: start y
        - prior_tensors[:, 3]: start x
        - prior_tensors[:, 4]: angle theta
        - prior_tensors[:, 5]: length
        - prior_tensors[:, 6:]: offsets
    - num_points: int, number of points to use for visualization (default 72)
    """
    # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates,
    # score[0] = negative prob, score[1] = positive prob
    # Convert image tensor to numpy for plotting
    image_np = np.ones((320, 800, 3))
    H, W = image_np.shape[:2]

    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    prior_tensors = prior_tensors.detach().cpu()
    for prior in prior_tensors:
        y = prior[2].item()
        x = prior[3].item()
        theta = prior[4].item()
        theta = theta * math.pi
        # print(theta)
        # if theta < math.pi / 4:
        #     continue
        # Generate y coordinates
        y_coords = np.linspace(0, 1, num_points)
        x_coords = x * (W - 1) + ((
            (1 - y_coords - y) * H) / np.tan(theta + 1e-5))

        y_coords = (y_coords + y) * H

        # Generate x coordinates
        # Plot the lane prior
        plt.plot(x_coords, y_coords, '-', linewidth=2)
        # Plot start point
        plt.plot(x_coords[0], y_coords[0], 'ro', markersize=5)

    plt.xlim(0, W)
    plt.ylim(H, 0)
    plt.legend()
    plt.title('Lane Priors Visualization')
    plt.show()


@MODELS.register_module()
class CLRHead(nn.Module):

    def __init__(
        self,
        num_points: int = 72,
        prior_feat_channels: int = 64,
        fc_hidden_dim: int = 64,
        num_priors: int = 192,
        num_fc: int = 2,
        refine_layers: int = 3,
        sample_points: int = 36,
        img_w: int = 800,
        img_h: int = 320,
        num_classes: int = 5,
        cut_height: int = 0,
        cls_loss: dict | None = None,
        cls_loss_weight: float = 2.,
        xyt_loss_weight: float = 0.5,
        iou_loss_weight: float = 2.,
        seg_loss_weight: float = 1.,
        num_classes_seg: int = 5,
        train_cfg=None,
        test_cfg=None,
    ):
        super(CLRHead, self).__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.num_classes = num_classes
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim

        self.num_classes = num_classes_seg
        self.ignore_label = 255
        self.bg_weight = 0.4
        self.cut_height = cut_height
        self.cls_loss_weight = cls_loss_weight
        self.xyt_loss_weight = xyt_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.seg_loss_weight = seg_loss_weight

        if cls_loss is None:
            cls_loss = dict(type='FocalLoss', alpha=0.25, gamma=2.0)

        self.cls_criterion = MODELS.build(cls_loss)

        self.register_buffer(
            name='sample_x_indexs',
            tensor=(torch.linspace(
                0, 1, steps=self.sample_points, dtype=torch.float32) *
                    self.n_strips).long())
        self.register_buffer(
            name='prior_feat_ys',
            tensor=torch.flip(
                (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        self.register_buffer(
            name='prior_ys',
            tensor=torch.linspace(
                1, 0, steps=self.n_offsets, dtype=torch.float32))

        self.prior_feat_channels = prior_feat_channels
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings(
        )  # None, None
        self.register_buffer(name='priors', tensor=init_priors)
        self.register_buffer(
            name='priors_on_featmap', tensor=priors_on_featmap)

        # generate xys for feature map
        self.seg_decoder = SegDecoder(self.img_h, self.img_w, self.num_classes,
                                      self.prior_feat_channels,
                                      self.refine_layers)

        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)

        self.roi_gather = ROIGather(self.prior_feat_channels, self.num_priors,
                                    self.sample_points, self.fc_hidden_dim,
                                    self.refine_layers)

        self.reg_layers = nn.Linear(
            self.fc_hidden_dim, self.n_offsets + 1 + 2 +
            1)  # n offsets + 1 length + start_x + start_y + theta
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)

        weights = torch.ones(self.num_classes)
        weights[0] = self.bg_weight
        self.criterion = torch.nn.NLLLoss(
            ignore_index=self.ignore_label, weight=weights)

        # init the weights here
        self.init_weights()

    # function to init layer weights
    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

    def get_device(self) -> str:
        return next(self.parameters()).device.type

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        """Pool prior feature from feature map.

        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, H, W)
        """

        batch_size = batch_features.shape[0]

        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1)

        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
        grid = torch.cat((prior_xs, prior_ys), dim=-1)
        feature = F.grid_sample(
            batch_features, grid, align_corners=True).permute(0, 2, 1, 3)

        feature = feature.reshape(batch_size * num_priors,
                                  self.prior_feat_channels, self.sample_points,
                                  1)
        return feature

    def generate_priors_from_embeddings(self):
        predictions = self.prior_embeddings.weight  # (num_prop, 3)

        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates,
        # score[0] = negative prob, score[1] = positive prob
        priors = predictions.new_zeros(
            (self.num_priors, 2 + 2 + 2 + self.n_offsets),
            device=predictions.device)

        priors[:, 2:5] = predictions.clone()
        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) *
            (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(self.num_priors, 1) -
              priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)) *
             self.img_h / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(
                 1, self.n_offsets) * math.pi + 1e-5))) / (
                     self.img_w - 1)

        # init priors on feature map
        priors_on_featmap = priors.clone()[..., 6 + self.sample_x_indexs]

        return priors, priors_on_featmap

    def _init_prior_embeddings(self):
        # [start_y, start_x, theta] -> all normalize
        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8

        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0],
                              (i // 2) * strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.16 if i % 2 == 0 else 0.32)

        for i in range(left_priors_nums,
                       left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1],
                              ((i - left_priors_nums) // 4 + 1) *
                              bottom_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.2 * (i % 4 + 1))

        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) *
                strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.68 if i % 2 == 0 else 0.84)

    # forward function here
    def forward(self,
                inputs,
                batch_data_samples,
                training: bool = False,
                **kwargs):
        """Take pyramid features as input to perform Cross Layer Refinement and
        finally output the prediction lanes.

        Each feature is a 4D tensor.
        Args:
            x: input features (list[Tensor])
        Return:
            prediction_list: each layer's prediction result
            seg: segmentation result for auxiliary loss
        """
        batch_features = list(inputs[len(inputs) - self.refine_layers:])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]

        if training:
            self.priors, self.priors_on_featmap = \
                self.generate_priors_from_embeddings()

        # visualize_lane_priors_tensor(self.priors, num_points=72)
        priors, priors_on_featmap = self.priors.repeat(
            batch_size, 1, 1), self.priors_on_featmap.repeat(batch_size, 1, 1)
        predictions_lists = []

        # iterative refine
        prior_features_stages = []
        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])

            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs)
            prior_features_stages.append(batch_prior_features)

            fc_features = self.roi_gather(prior_features_stages,
                                          batch_features[stage], stage)

            fc_features = fc_features.view(num_priors, batch_size,
                                           -1).reshape(batch_size * num_priors,
                                                       self.fc_hidden_dim)

            cls_features = fc_features.clone()
            reg_features = fc_features.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)

            cls_logits = self.cls_layers(cls_features)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[1])  # (B, num_priors, 2)
            reg = reg.reshape(batch_size, -1, reg.shape[1])

            predictions = priors.clone()
            predictions[:, :, :2] = cls_logits

            predictions[:, :,
                        2:5] += reg[:, :, :3]  # also reg theta angle here
            predictions[:, :, 5] = reg[:, :, 3]  # length

            def tran_tensor(t):
                return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)

            predictions[..., 6:] = (
                tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
                ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
                  tran_tensor(predictions[..., 2])) * self.img_h /
                 torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))
            ) / (
                self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]

            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        if training:
            seg = None
            seg_features = torch.cat([
                F.interpolate(
                    feature,
                    size=[
                        batch_features[-1].shape[2],
                        batch_features[-1].shape[3]
                    ],
                    mode='bilinear',
                    align_corners=False) for feature in batch_features
            ],
                                     dim=1)
            seg = self.seg_decoder(seg_features)
            output = {'predictions_lists': predictions_lists, 'seg': seg}
            return output, batch_data_samples

        return predictions_lists[-1]

    def predictions_to_pred(self, predictions, scores, org_width):
        """Convert predictions to internal Lane structure for evaluation."""
        self.prior_ys = self.prior_ys.to(predictions.device)
        self.prior_ys = self.prior_ys.double()

        lanes = []
        for lane, score in zip(predictions, scores, strict=True):
            lane_xs = lane[6:]  # normalized value
            start = min(
                max(0, int(round(lane[2].item() * self.n_strips))),
                self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool_))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            lane_ys = (lane_ys * (org_width - self.cut_height) +
                       self.cut_height) / org_width
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)
            lane_prediction = Lane(
                points=points.cpu().numpy(),
                metadata={
                    'start_x': lane[3].cpu().item(),
                    'start_y': lane[2].cpu().item(),
                    'conf': score.cpu().item()
                })
            lanes.append(lane_prediction)
        return lanes

    def loss(self, x, batch, training: bool = True, **kwargs):
        output, batch = self.forward(
            x, batch_data_samples=batch, training=training)
        predictions_lists = output['predictions_lists']
        targets = torch.stack([torch.tensor(b.lines_array)
                               for b in batch]).clone()
        targets = targets.to(x[0].device)

        target_batch_seg = torch.stack(
            [b.gt_sem_seg.sem_seg.squeeze(0) for b in batch]).clone().long()
        target_batch_seg = target_batch_seg.to(x[0].device)
        cls_loss = 0
        reg_xytl_loss = 0
        iou_loss = 0

        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets):
                target = target[target[:, 1] == 1]

                if len(target) == 0:
                    # If there are no targets, all predictions have to be
                    # negatives (i.e., 0 confidence)
                    cls_target = predictions.new_zeros(
                        predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + self.cls_criterion(
                        cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h)

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]

                # regression targets -> [start_y, start_x, theta]
                # (all transformed to absolute values), only on matched pairs
                reg_yxtl = predictions[matched_row_inds, 2:6]
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= (self.img_w - 1)
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                target_yxtl = target[matched_col_inds, 2:6].clone()

                # regression targets -> S coordinates
                # (all transformed to absolute values)
                reg_pred = predictions[matched_row_inds, 6:]
                reg_pred *= (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()

                with torch.no_grad():
                    predictions_starts = torch.clamp(
                        (predictions[matched_row_inds, 2] *
                         self.n_strips).round().long(), 0, self.n_strips
                    )  # ensure the predictions starts is valid
                    target_starts = (target[matched_col_inds, 2] *
                                     self.n_strips).round().long()
                    target_yxtl[:, -1] -= (predictions_starts - target_starts
                                           )  # reg length

                # Loss calculation
                cls_loss = cls_loss + self.cls_criterion(
                    cls_pred, cls_target).sum() / target.shape[0]

                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 2] *= 180
                reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(
                    reg_yxtl, target_yxtl, reduction='none').mean()

                iou_loss = iou_loss + liou_loss(
                    reg_pred, reg_targets, self.img_w, length=15)

        # extra segmentation loss
        seg_loss = self.criterion(
            F.log_softmax(output['seg'], dim=1), target_batch_seg)

        cls_loss /= (len(targets) * self.refine_layers)
        reg_xytl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)

        loss = cls_loss * self.cls_loss_weight \
            + reg_xytl_loss * self.xyt_loss_weight \
            + seg_loss * self.seg_loss_weight \
            + iou_loss * self.iou_loss_weight

        return_value = {
            'loss': loss,
            'cls_loss': cls_loss * self.cls_loss_weight,
            'reg_xytl_loss': reg_xytl_loss * self.xyt_loss_weight,
            'seg_loss': seg_loss * self.seg_loss_weight,
            'iou_loss': iou_loss * self.iou_loss_weight
        }
        return return_value

    def get_lanes(self,
                  output,
                  org_widths,
                  as_lanes=True,
                  conf_threshold=0.05,
                  nms_threshold=0.5,
                  top_k=5):
        """Convert model output to lanes."""
        softmax = nn.Softmax(dim=1)

        decoded = []
        for predictions, org_width in zip(output, org_widths, strict=True):
            # filter out the conf lower than conf threshold
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= conf_threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            if nms is not None and self.get_device() != 'cpu':
                nms_predictions = predictions.detach().clone()
                nms_predictions = torch.cat(
                    [nms_predictions[..., :4], nms_predictions[..., 5:]],
                    dim=-1)
                nms_predictions[...,
                                4] = nms_predictions[..., 4] * self.n_strips
                nms_predictions[..., 5:] = nms_predictions[..., 5:] * (
                    self.img_w - 1)

                keep, num_to_keep, _ = nms(
                    nms_predictions,
                    scores,
                    overlap=nms_threshold,
                    top_k=top_k)
                keep = keep[:num_to_keep]
                predictions = predictions[keep]
                scores = scores[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions, scores, org_width)
            else:
                pred = predictions
            decoded.append(pred)

        return decoded

    def predict(self, x, batched_data_samples, *args, **kwargs):
        """Perform inference on the input image.

        Args:
            x: input image
        Return:
            lanes: list of lanes
        """
        output = self.forward(x, batched_data_samples)
        org_widths, org_heights = zip(*[(sample.org_width, sample.org_height)
                                        for sample in batched_data_samples])
        batched_lane = self.get_lanes(output, org_widths, as_lanes=True)
        preds = []
        for lanes, org_width, org_height in zip(batched_lane, org_widths,
                                                org_heights):
            instance = InstanceData()
            instance_points = []
            for lane in lanes:
                points = lane.points
                points[..., 0] *= (org_width - 1)
                points[..., 1] *= (org_height - 1)
                instance_points.append(points)
            instance.lines = instance_points
            instance.line_conf = [lane.metadata['conf'] for lane in lanes]

            preds.append(instance)
        return preds
