import pytest
import torch
from mmengine.config import ConfigDict

from mmdet.registry import MODELS
from mmdet.testing import get_detector_cfg
from mmdet.utils import register_all_modules

register_all_modules()


def make_batch_data_samples(num_targets_per_sample,
                            img_shape=(800, 1200),
                            device='cpu'):
    """Helper to create mock batch_data_samples."""
    from mmengine.structures import InstanceData

    from mmdet.structures import DetDataSample

    batch_data_samples = []
    for num_targets in num_targets_per_sample:
        sample = DetDataSample()
        sample.set_metainfo({
            'img_shape': img_shape,
            'batch_input_shape': img_shape
        })
        gt_instances = InstanceData()
        if num_targets > 0:
            bboxes = torch.rand(num_targets, 4)
            bboxes[:, 2:] = bboxes[:, 2:] + bboxes[:, :2]
            bboxes = bboxes.clamp(0, 1)
            img_w, img_h = img_shape[1], img_shape[0]
            bboxes[:, 0::2] *= img_w
            bboxes[:, 1::2] *= img_h
            gt_instances.bboxes = bboxes.to(device)
            gt_instances.labels = torch.randint(
                0, 80, (num_targets, ), device=device)
        else:
            gt_instances.bboxes = torch.zeros((0, 4), device=device)
            gt_instances.labels = torch.zeros((0, ),
                                              dtype=torch.long,
                                              device=device)
        sample.gt_instances = gt_instances
        batch_data_samples.append(sample)
    return batch_data_samples


def build_dino_detector(embed_dims=256, neck_out_channels=None):
    """Build a minimal DINO detector.

    Uses a lightweight ResNet-18 backbone so no pretrained weights are needed.
    ``neck_out_channels`` defaults to ``embed_dims``; pass a different value to
    simulate a misconfigured neck (issue #25).
    """
    if neck_out_channels is None:
        neck_out_channels = embed_dims

    cfg = get_detector_cfg('dino/dino-4scale_r50_8xb2-12e_coco.py')

    # Swap ResNet-50 (needs pretrained) for ResNet-18 (no pretrained needed)
    cfg.backbone = ConfigDict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch')
    cfg.neck.in_channels = [128, 256, 512]  # ResNet-18 stage channels
    cfg.neck.out_channels = neck_out_channels

    # Keep all transformer dims consistent with embed_dims
    cfg.encoder.layer_cfg.self_attn_cfg.embed_dims = embed_dims
    cfg.encoder.layer_cfg.ffn_cfg.embed_dims = embed_dims
    cfg.encoder.layer_cfg.ffn_cfg.feedforward_channels = embed_dims * 4
    cfg.decoder.layer_cfg.self_attn_cfg.embed_dims = embed_dims
    cfg.decoder.layer_cfg.cross_attn_cfg.embed_dims = embed_dims
    cfg.decoder.layer_cfg.ffn_cfg.embed_dims = embed_dims
    cfg.decoder.layer_cfg.ffn_cfg.feedforward_channels = embed_dims * 4
    cfg.positional_encoding.num_feats = embed_dims // 2
    cfg.bbox_head.embed_dims = embed_dims

    return MODELS.build(cfg)


def make_neck_feats(batch_size, neck_out_channels, num_levels=4):
    """Simulate multi-scale feature maps from a neck with given
    out_channels."""
    spatial_shapes = [(25, 34), (13, 17), (7, 9), (4, 5)]
    return [
        torch.randn(batch_size, neck_out_channels, h, w)
        for h, w in spatial_shapes[:num_levels]
    ]


class TestDinoEmbedDimsConsistency:
    """
    Confirms https://github.com/VBTI-development/onedl-mmdetection/issues/25

    DINO requires embed_dims to be consistent across:
      1. The neck's out_channels (e.g. ChannelMapper)
      2. The DINO transformer encoder and decoder embed_dims
      3. CdnQueryGenerator embed_dims (set from the detector's embed_dims)

    With the default config (embed_dims=256, neck.out_channels=256) everything
    works.  If the backbone or neck is changed so that the neck output channels
    differ from the transformer embed_dims, a RuntimeError is raised inside
    ``DINO.pre_transformer``, which concatenates positional embeddings (sized
    embed_dims) with the neck feature maps (sized neck_out_channels).
    """

    # ------------------------------------------------------------------ #
    # 1. Default config — neck=256, embed_dims=256 — must work            #
    # ------------------------------------------------------------------ #

    def test_default_256_config_works(self):
        """Default DINO config: neck out_channels=256, embed_dims=256."""
        batch_size = 2
        model = build_dino_detector(embed_dims=256)
        mlvl_feats = make_neck_feats(batch_size, neck_out_channels=256)
        batch_data_samples = make_batch_data_samples([3, 5])

        # Must not raise
        enc_inputs, dec_inputs = model.pre_transformer(mlvl_feats,
                                                       batch_data_samples)
        assert enc_inputs is not None

    # ------------------------------------------------------------------ #
    # 2. Neck changed to 512, embed_dims still 256 — shape mismatch        #
    # ------------------------------------------------------------------ #

    def test_neck_512_embed_dims_256_raises_shape_mismatch(self):
        """Neck out_channels changed to 512 (e.g. wider backbone), but the
        transformer embed_dims left at 256.

        DINO.pre_transformer tries to broadcast positional embeddings (256-dim)
        onto 512-dim neck features, raising a RuntimeError.
        """
        batch_size = 2
        model = build_dino_detector(embed_dims=256, neck_out_channels=256)
        mlvl_feats = make_neck_feats(batch_size, neck_out_channels=512)

        with pytest.raises(RuntimeError):
            model.pre_transformer(mlvl_feats, make_batch_data_samples([3, 5]))

    # ------------------------------------------------------------------ #
    # 3. Neck changed to 128, embed_dims still 256 — shape mismatch        #
    # ------------------------------------------------------------------ #

    def test_neck_128_embed_dims_256_raises_shape_mismatch(self):
        """Neck out_channels changed to 128 (e.g. lightweight backbone), but
        the transformer embed_dims left at 256."""
        batch_size = 2
        model = build_dino_detector(embed_dims=256, neck_out_channels=256)
        mlvl_feats = make_neck_feats(batch_size, neck_out_channels=128)

        with pytest.raises(RuntimeError):
            model.pre_transformer(mlvl_feats, make_batch_data_samples([2, 4]))

    # ------------------------------------------------------------------ #
    # 4. Both neck and transformer dims updated consistently — no mismatch #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize('embed_dims', [128, 256, 512])
    def test_consistent_neck_and_transformer_dims_works(self, embed_dims):
        """When neck out_channels and every transformer embed_dims are set to
        the same value, no shape error occurs inside pre_transformer."""
        batch_size = 2
        model = build_dino_detector(embed_dims=embed_dims)
        mlvl_feats = make_neck_feats(batch_size, neck_out_channels=embed_dims)
        batch_data_samples = make_batch_data_samples([2, 3])

        # Must not raise
        enc_inputs, dec_inputs = model.pre_transformer(mlvl_feats,
                                                       batch_data_samples)
        assert enc_inputs is not None

    @pytest.mark.parametrize('embed_dims', [128, 256, 512])
    def test_full_forward_transformer_consistent_dims_works(self, embed_dims):
        """Validates the fix to DinoTransformerDecoder (issue #25).

        Before the fix, ``coordinate_to_encoding`` used a hardcoded
        ``num_feats=128``, producing 512-dim output regardless of
        ``embed_dims``, while ``ref_point_head`` was built as
        ``MLP(embed_dims * 2, ...)``.  The two sizes only matched by
        coincidence when ``embed_dims == 256`` (4 * 128 == 256 * 2).
        For any other value the decoder raised a RuntimeError.

        After the fix, ``num_feats=embed_dims`` and
        ``ref_point_head = MLP(embed_dims * 4, ...)`` stay in sync.
        This test exercises the full encoder + decoder path so the
        shape contract inside the decoder is actually checked.
        """
        batch_size = 2
        model = build_dino_detector(embed_dims=embed_dims)
        model.eval()

        mlvl_feats = make_neck_feats(batch_size, neck_out_channels=embed_dims)
        batch_data_samples = make_batch_data_samples([2, 3])

        # forward_transformer runs pre_transformer → encoder → pre_decoder
        # → decoder, exercising the ref_point_head MLP inside the decoder.
        with torch.no_grad():
            results = model.forward_transformer(mlvl_feats, batch_data_samples)
        assert results is not None

    # ------------------------------------------------------------------ #
    # 5. CdnQueryGenerator embed_dims matches detector embed_dims          #
    # ------------------------------------------------------------------ #

    def test_cdn_query_generator_embed_dims_matches_detector(self):
        """The DINO detector sets CdnQueryGenerator.embed_dims ==
        self.embed_dims at construction time.

        dn_label_query's last dimension must equal embed_dims so the decoder
        can concatenate CDN and matching queries.
        """
        for embed_dims in [128, 256, 512]:
            model = build_dino_detector(embed_dims=embed_dims)
            batch_data_samples = make_batch_data_samples([3, 5])

            dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
                model.dn_query_generator(batch_data_samples)

            assert dn_label_query.shape[-1] == embed_dims, (
                f'embed_dims={embed_dims}: dn_label_query last dim '
                f'{dn_label_query.shape[-1]} != detector embed_dims')

    # ------------------------------------------------------------------ #
    # 6. Attention mask total size matches num_queries + num_dn_queries   #
    # ------------------------------------------------------------------ #

    def test_attn_mask_size_consistent_with_num_matching_queries(self):
        """The attention mask produced by CdnQueryGenerator must have total
        size (num_dn_queries + num_matching_queries).

        A mismatch here would cause the decoder's multi-head attention to fail
        at broadcast time.
        """
        model = build_dino_detector(embed_dims=256)
        batch_data_samples = make_batch_data_samples([3, 5])

        dn_label_query, _, attn_mask, dn_meta = \
            model.dn_query_generator(batch_data_samples)

        num_dn = dn_meta['num_denoising_queries']
        num_matching = model.num_queries

        expected_total = num_dn + num_matching
        assert attn_mask.shape == (expected_total, expected_total), (
            f'attn_mask shape {attn_mask.shape} != '
            f'({expected_total}, {expected_total}). '
            f'num_dn={num_dn}, num_matching={num_matching}')
