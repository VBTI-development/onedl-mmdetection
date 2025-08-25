custom_imports = dict(imports=[], allow_failed_imports=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOneDLSegmentation'),
    dict(
        type='ImageAug',
        pipeline=[
            dict(
                name='Resize',
                p=1.0,
                parameters=dict(size=dict(height=320, width=800))),
            dict(name='HorizontalFlip', p=0.5, parameters=dict(p=1.0)),
            dict(name='ChannelShuffle', p=0.1, parameters=dict(p=1.0)),
            dict(
                name='MultiplyAndAddToBrightness',
                p=0.6,
                parameters=dict(add=[-10, 10], mul=[0.85, 1.15])),
            dict(
                name='AddToHueAndSaturation',
                p=0.7,
                parameters=dict(value=[-10, 10])),
            dict(
                name='OneOf',
                p=0.2,
                transforms=[
                    dict(name='MotionBlur', parameters=dict(k=[3, 5])),
                    dict(name='MedianBlur', parameters=dict(k=[3, 5]))
                ]),
            dict(
                name='Affine',
                p=0.7,
                parameters=dict(
                    rotate=[-10, 10],
                    scale=[0.8, 1.2],
                    translate_percent=dict(x=[-0.1, 0.1], y=[-0.1, 0.1])))
        ]),
    dict(
        type='LinesToArray',
        img_height=320,
        img_width=800,
        max_lines=4,
        num_points=72),
    dict(type='PackLineDetectionInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ImageAug',
        pipeline=[
            dict(
                name='Resize',
                p=1.0,
                parameters=dict(size=dict(height=320, width=800)))
        ]),
    dict(
        type='LinesToArray',
        img_height=320,
        img_width=800,
        max_lines=4,
        num_points=72),
    dict(type='PackLineDetectionInputs')
]

# Model configuration
model = dict(
    type='CLRNet',
    data_preprocessor=dict(
        type='LineDetDataProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='CLRFPN',
        in_channels=[256, 512, 1024],
        out_channels=64,
        num_outs=3,
        attention=False),
    head=dict(
        type='CLRHead',
        num_priors=192,
        refine_layers=3,
        fc_hidden_dim=64,
        sample_points=36))

# Optimizer configuration
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0)
        }))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=1,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True)
]

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Evaluation configuration
val_evaluator = dict(type='CULaneMetric')
test_evaluator = dict(type='CULaneMetric')

# Hook configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best='auto',
        greater_keys=['culane/TP_50']),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmengine'),
    visualization=dict(type='DetVisualizationHook', show=False))

# Environment configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# Visualization configuration
visualizer = dict(
    type='DetLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')])

# Log configuration
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# Other configurations
default_scope = 'mmdet'
gpu_ids = (0, )
randomness = dict(seed=4389)
log_level = 'INFO'
load_from = None
resume = False
