# dataset settings
# D3 in the config name means the whole dataset is divided into 3 folds
# We only use one fold for efficient experiments
dataset_type = 'WaymoDataset'
data_root = '/scratch/cmpe295-wu/hj/waymo_data/waymo_1_4_1/part_3/kitti_format/'
class_names = ['Car', 'Pedestrian', 'Cyclist']
input_modality = dict(use_lidar=False, use_camera=True)
metainfo = dict(CLASSES=class_names)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/waymo/kitti_format/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
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
    dict(type='Pack3DDetInputs', keys=['img']),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='mmdet.Resize', scale_factor=1.0),
    dict(type='Pack3DDetInputs', keys=['img']),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_train.pkl',
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_LEFT='training/image_1',
            CAM_FRONT_RIGHT='training/image_2',
            CAM_SIDE_LEFT='training/image_3',
            CAM_SIDE_RIGHT='training/image_4'),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='Camera',
        load_type='fov_image_based',
        # load one frame every five frames
        load_interval=5,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_LEFT='training/image_1',
            CAM_FRONT_RIGHT='training/image_2',
            CAM_SIDE_LEFT='training/image_3',
            CAM_SIDE_RIGHT='training/image_4'),
        ann_file='waymo_infos_val.pkl',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='Camera',
        load_type='fov_image_based',
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_LEFT='training/image_1',
            CAM_FRONT_RIGHT='training/image_2',
            CAM_SIDE_LEFT='training/image_3',
            CAM_SIDE_RIGHT='training/image_4'),
        ann_file='waymo_infos_val.pkl',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='Camera',
        load_type='fov_image_based',
        backend_args=backend_args))

val_evaluator = dict(
    type='WaymoMetric',
    ann_file='/scratch/cmpe295-wu/hj/waymo_data/waymo_1_4_1/part_3/kitti_format/waymo_infos_val.pkl',
    waymo_bin_file='/scratch/cmpe295-wu/hj/waymo_data/waymo_1_4_1/fov_gt_all.bin',
    data_root='/scratch/cmpe295-wu/hj/waymo_data/waymo_1_4_1/part_3/waymo_format',
    metric='LET_mAP',
    load_type='fov_image_based',
    backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
