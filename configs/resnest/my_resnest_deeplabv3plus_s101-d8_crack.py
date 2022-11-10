norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnest101',
    backbone=dict(
        type='ResNeSt',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=1.0), dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=1.0)]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=1.0), dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=1.0)]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'PascalVOCDataset'
data_root = '/home/x1/Documents/人工智能CV+NLP技术(项目实战课)/第5期/第四章-MMLAB实战系列/mmsegmentation-0.25.0/data/CrackForest/crack_voc'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (480, 320)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(480, 320), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='PascalVOCDataset',
            data_root='/home/x1/Documents/人工智能CV+NLP技术(项目实战课)/第5期/第四章-MMLAB实战系列/mmsegmentation-0.25.0/data/CrackForest/crack_voc',
            img_dir='JPEGImages',
            ann_dir='SegmentationClassPNG',
            split='train.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize', img_scale=(480, 320), ratio_range=(0.5, 2.0)),
                dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ])),
    val=dict(
        type='PascalVOCDataset',
        data_root='/home/x1/Documents/人工智能CV+NLP技术(项目实战课)/第5期/第四章-MMLAB实战系列/mmsegmentation-0.25.0/data/CrackForest/crack_voc',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassPNG',
        split='val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(480, 320),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='PascalVOCDataset',
        data_root='/home/x1/Documents/人工智能CV+NLP技术(项目实战课)/第5期/第四章-MMLAB实战系列/mmsegmentation-0.25.0/data/CrackForest/crack_voc',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassPNG',
        split='test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(480, 320),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=2500)
checkpoint_config = dict(by_epoch=False, interval=250)
evaluation = dict(interval=250, metric='mIoU', pre_eval=True)
work_dir = './work_dirs/my_resnest_deeplabv3plus_s101-d8_crack'
gpu_ids = [0]
auto_resume = False
