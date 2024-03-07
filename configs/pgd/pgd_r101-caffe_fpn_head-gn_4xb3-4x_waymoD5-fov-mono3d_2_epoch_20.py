_base_ = [
    '../_base_/datasets/waymoD5-fov-mono3d-3class_2.py',
    '../_base_/models/pgd.py', '../_base_/schedules/mmdet-schedule-1x.py',
    '../_base_/default_runtime.py'
]

load_from = '/home/001891254/295a_project/CMPE295_mmdetection3d/global_weight_epoch_20.pth'

num_workers=1 
batch_size=1
# model settings
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(frozen_stages=0),
    neck=dict(start_level=1, num_outs=3),
    bbox_head=dict(
        num_classes=3,
        bbox_code_size=7,
        pred_attrs=False,
        pred_velo=False,
        pred_bbox2d=True,
        use_onlyreg_proj=True,
        strides=(8, 16, 32),
        regress_ranges=((-1, 128), (128, 256), (256, 1e8)),
        group_reg_dims=(2, 1, 3, 1, 16,
                        4),  # offset, depth, size, rot, kpts, bbox2d
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
            (256, ),  # kpts
            (256, )  # bbox2d
        ),
        centerness_branch=(256, ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        use_depth_classifier=True,
        depth_branch=(256, ),
        depth_range=(0, 70),
        depth_unit=10,
        division='uniform',
        depth_bins=8,
        pred_keypoints=True,
        weight_dim=1,
        loss_depth=dict(
            type='UncertainSmoothL1Loss', alpha=1.0, beta=3.0,
            loss_weight=1.0),
        bbox_coder=dict(
            type='PGDBBoxCoder',
            base_depths=((28.01, 16.32), ),
            base_dims=((0.8, 1.73, 0.6), (1.76, 1.73, 0.6), (3.9, 1.56, 1.6)),
            #base_depths=(
            #    (1.7, 0.5),  # Pedestrian (average height, average depth)
            #    (1.7, 1.5),  # Cyclist (average height, average depth including the bike)
            #    (1.6, 3.5),  # Car (average height, average depth)
            #    (2.5, 5.0),  # Van (average height, average depth)
            #    (3.5, 7.0),  # Truck (average height, average depth)
            #    (1.3, 0.5),  # Person_sitting (average height while sitting, average depth)
            #    (3.5, 10.0), # Tram (average height, average depth)
            #    (2.0, 3.0),  # Misc (a generic average assuming it's a mix of objects)
            #    (1.5, 2.5)   # DontCare (an average considering it could be any object)
            #),
            #base_dims=(
            #    (0.5, 0.5, 1.7),  # Pedestrian (width, depth, height)
            #    (0.6, 1.7, 1.7),  # Cyclist
            #    (1.8, 4.5, 1.6),  # Car
            #    (2.0, 5.2, 2.5),  # Van
            #    (2.5, 8.0, 3.5),  # Truck
            #    (0.8, 0.6, 1.3),  # Person_sitting
            #    (2.5, 11.0, 3.5), # Tram
            #    (1.8, 3.0, 2.0),  # Misc
            #    (1.8, 2.8, 1.5)   # DontCare
            #),
            code_size=7)),
    # set weight 1.0 for base 7 dims (offset, depth, size, rot)
    # 0.2 for 16-dim keypoint offsets and 1.0 for 4-dim 2D distance targets
    train_cfg=dict(code_weight=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0
    ]),
    test_cfg=dict(nms_pre=100, nms_thr=0.05, score_thr=0.001, max_per_img=20))

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    # base shape (1248, 832), scale (0.95, 1.05)
    dict(type='mmdet.Resize', scale_factor=1.0),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='mmdet.Resize', scale_factor=1.0),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=batch_size, num_workers=num_workers, dataset=dict(pipeline=train_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2)
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1)
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

train_cfg = dict(max_epochs=5, val_interval=5)
auto_scale_lr = dict(base_batch_size=8)