_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
    './dino-4scale_r50_8xb2-12e_coco.py'
]
norm_cfg = dict(type='LN2d', requires_grad=True)

model = dict(
    type='DINO',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=5,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='DINOv3ConvNextBackbone',
        finetuning=True,
        layers=[0,1,2,3],
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[96, 192, 384, 768],
        kernel_size=1,
        out_channels=384,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=384, num_levels=5,
                               dropout=0.1),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=384,
                feedforward_channels=1024,  # 1024 for DeformDETR
                ffn_drop=0.1))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=384, num_heads=8,
                               dropout=0.1),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=384, num_levels=5,
                                dropout=0.1),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=384,
                feedforward_channels=1024,  # 1024 for DeformDETR
                ffn_drop=0.1)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=384 // 2,
        normalize=True,
        offset=-0.5,  # -0.5 for DeformDETR
        temperature=10000),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        embed_dims=384,
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300)
)