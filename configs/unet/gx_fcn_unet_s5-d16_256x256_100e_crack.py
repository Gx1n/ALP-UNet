norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=128,
    #     in_index=3,
    #     channels=64,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=dict(type='SyncBN', requires_grad=True),
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))
dataset_type = 'PascalVOCDataset'
data_root = '/home/x1/Documents/人工智能CV+NLP技术(项目实战课)/第5期/第四章-MMLAB实战系列/mmsegmentation-0.25.0/data/CrackForest/crack_voc'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (480, 320)
crop_size = (256, 256)
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
                    type='Resize',
                    img_scale=(480, 320),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
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
# log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]
# cudnn_benchmark = True
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
# # lr_config = dict(
# #     policy='step',
# #     warmup='linear',
# #     warmup_iters=500,
# #     warmup_ratio=0.001,
# #     step=[16, 22])
# runner = dict(type='EpochBasedRunner', max_epochs=200)
# checkpoint_config = dict(by_epoch=True, interval=50)
# evaluation = dict(interval=10, metric='mIoU', pre_eval=True)
# work_dir = './work_dirs/gx_fcn_unet_s5-d16_256x256_200e_crack'
# gpu_ids = [0]
# auto_resume = False
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
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 22])
#runner = dict(type='EpochBasedRunner', max_epochs=200)
runner = dict(type='IterBasedRunner', max_iters=2500)
checkpoint_config = dict(by_epoch=False, interval=250)
evaluation = dict(interval=250, metric='mIoU', pre_eval=True)
work_dir = './work_dirs/gx_fcn_unet_s5-d16_256x256_100e_crack'
gpu_ids = [0]
auto_resume = False
