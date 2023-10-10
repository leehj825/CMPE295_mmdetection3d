# CMPE295_Project

Start from https://github.com/open-mmlab/mmdetection3d

## Models and Code Bases
- **Model**: PGD (FCOS3D++)

- **MMDetection3D**: https://github.com/open-mmlab/mmdetection3d

## Dataset Preparation
Kitti Dataset for 3D Object Detection: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

Waymo Dataset will be used with converting to Kitti format

### Kitti Data Format
Refer: https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html
```
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
```

### Script to convert to COCO format annotation
```
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```
3 Classes are defined in annotation files.
```
+------------+--------+
| category   | number |
+------------+--------+
| Pedestrian | 2207   |
| Cyclist    | 734    |
| Car        | 14352  |
+------------+--------+
```
## Training
### PGD config
#### Train from scratch: configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d_2workers.py
```
num_workers=2 
batch_size=2
:
:

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
        save_best='auto',
        max_keep_ckpts=10))
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
        end=10,
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1)
]


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1))

train_cfg = dict(max_epochs=10, val_interval=2)
auto_scale_lr = dict(base_batch_size=8)
```
The training script is provided under tools folder.
```
> python tools/train.py configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d_2workers.py
```
#### Re-Training: pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d_tune.py
HPC server often aborts before completing 10x epoches.  Loading from last saved checkpoint can resume the training.
```
load_from = 'work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d_tune/20231009_023518/epoch_7.pth'
```
Training output
```
2023/10/08 11:07:02 - mmengine - INFO - Checkpoints will be saved to /home/001891254/CMPE295_mmdetection3d/mmdetection3d/work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d_2workers.
2023/10/08 11:07:05 - mmengine - INFO - Epoch(train)  [1][   1/1856]  base_lr: 3.3333e-05 lr: 3.3333e-05  eta: 12:22:17  time: 2.3998  data_time: 0.2418  memory: 5761  grad_norm: 210.8646  loss: 38.2627  loss_cls: 1.0523  loss_offset: 1.4213  loss_size: 2.2083  loss_rotsin: 0.7576  loss_dir: 0.6425  loss_depth: 5.5293  loss_kpts: 0.8714  loss_bbox2d: 23.6761  loss_consistency: 1.3716  loss_centerness: 0.7322
2023/10/08 11:07:06 - mmengine - INFO - Epoch(train)  [1][   2/1856]  base_lr: 3.3467e-05 lr: 3.3467e-05  eta: 8:34:38  time: 1.6639  data_time: 0.1278  memory: 6177  grad_norm: 247.7931  loss: 43.3232  loss_cls: 1.0396  loss_offset: 1.4222  loss_size: 2.8433  loss_rotsin: 0.8032  loss_dir: 0.6629  loss_depth: 10.4461  loss_kpts: 0.8357  loss_bbox2d: 23.0719  loss_consistency: 1.4819  loss_centerness: 0.7163
2023/10/08 11:07:07 - mmengine - INFO - Epoch(train)  [1][   3/1856]  base_lr: 3.3601e-05 lr: 3.3601e-05  eta: 7:18:05  time: 1.4165  data_time: 0.0887  memory: 6177  grad_norm: 239.9165  loss: 42.9269  loss_cls: 1.0215  loss_offset: 1.4239  loss_size: 2.9068  loss_rotsin: 0.7887  loss_dir: 0.6898  loss_depth: 11.0124  loss_kpts: 0.7610  loss_bbox2d: 22.1248  loss_consistency: 1.4944  loss_centerness: 0.7035
2023/10/08 11:07:08 - mmengine - INFO - Epoch(train)  [1][   4/1856]  base_lr: 3.3734e-05 lr: 3.3734e-05  eta: 6:39:56  time: 1.2932  data_time: 0.0693  memory: 6198  grad_norm: 206.2112  loss: 41.7868  loss_cls: 1.0047  loss_offset: 1.4236  loss_size: 2.5093  loss_rotsin: 0.7979  loss_dir: 0.6958  loss_depth: 9.5202  loss_kpts: 0.7677  loss_bbox2d: 22.9548  loss_consistency: 1.4222  loss_centerness: 0.6906
2023/10/08 11:07:09 - mmengine - INFO - Epoch(train)  [1][   5/1856]  base_lr: 3.3868e-05 lr: 3.3868e-05  eta: 6:17:02  time: 1.2192  data_time: 0.0577  memory: 6194  grad_norm: 182.8207  loss: 41.2189  loss_cls: 0.9842  loss_offset: 1.4163  loss_size: 2.2555  loss_rotsin: 0.7389  loss_dir: 0.6991  loss_depth: 8.3291  loss_kpts: 0.7911  loss_bbox2d: 23.9388  loss_consistency: 1.3850  loss_centerness: 0.6811
2023/10/08 11:07:09 - mmengine - INFO - Epoch(train)  [1][   6/1856]  base_lr: 3.4001e-05 lr: 3.4001e-05  eta: 6:01:52  time: 1.1702  data_time: 0.0500  memory: 6182  grad_norm: 176.1497  loss: 39.7869  loss_cls: 0.9560  loss_offset: 1.4128  loss_size: 1.9899  loss_rotsin: 0.6785  loss_dir: 0.7006  loss_depth: 7.8221  loss_kpts: 0.7435  loss_bbox2d: 23.4191  loss_consistency: 1.3936  loss_centerness: 0.6708
2023/10/08 11:07:10 - mmengine - INFO - Epoch(train)  [1][   7/1856]  base_lr: 3.4135e-05 lr: 3.4135e-05  eta: 5:51:00  time: 1.1351  data_time: 0.0447  memory: 6233  grad_norm: 164.5725  loss: 39.2791  loss_cls: 0.9361  loss_offset: 1.4112  loss_size: 1.8473  loss_rotsin: 0.6567  loss_dir: 0.7045  loss_depth: 7.2114  loss_kpts: 0.7349  loss_bbox2d: 23.7661  loss_consistency: 1.3472  loss_centerness: 0.6637
2023/10/08 11:07:11 - mmengine - INFO - Epoch(train)  [1][   8/1856]  base_lr: 3.4269e-05 lr: 3.4269e-05  eta: 5:42:46  time: 1.1086  data_time: 0.0405  memory: 6177  grad_norm: 158.1040  loss: 38.7946  loss_cls: 0.8901  loss_offset: 1.4160  loss_size: 1.7141  loss_rotsin: 0.6028  loss_dir: 0.7213  loss_depth: 6.6318  loss_kpts: 0.7264  loss_bbox2d: 24.1232  loss_consistency: 1.3116  loss_centerness: 0.6574
2023/10/08 11:07:12 - mmengine - INFO - Epoch(train)  [1][   9/1856]  base_lr: 3.4402e-05 lr: 3.4402e-05  eta: 5:36:23  time: 1.0880  data_time: 0.0373  memory: 6177  grad_norm: 165.4552  loss: 37.0609  loss_cls: 0.8523  loss_offset: 1.4144  loss_size: 1.6091  loss_rotsin: 0.5995  loss_dir: 0.7442  loss_depth: 6.2100  loss_kpts: 0.7062  loss_bbox2d: 22.9726  loss_consistency: 1.2927  loss_centerness: 0.6598
```

<img width="922" alt="image" src="https://github.com/leehj825/CMPE295_mmdetection3d/assets/21224335/4607003b-126a-49ba-b932-1a46a888ead3">

## Reference
tools/mono_det_demo.py is modified to draw 3D bounding boxes from the prediction and ground-truth.
```
> python demo/mono_det_demo.py demo/data/kitti/000008.png demo/data/kitti/000008.pkl configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d_tune.py work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d_tune/20231008_172104/epoch_7.pth --cam-type CAM2 --out-dir demo/output.png
```
### Ground Truth
![image](https://github.com/leehj825/CMPE295_mmdetection3d/assets/21224335/9f5c4258-b7df-4c18-879f-3dd075a1245a)
### Epoch #4
![image](https://github.com/leehj825/CMPE295_mmdetection3d/assets/21224335/33cbf55f-08c3-490a-8008-c1438ab393cb)
### Epoch #11
![image](https://github.com/leehj825/CMPE295_mmdetection3d/assets/21224335/f04405ef-5512-4826-959b-59eead9e3fee)
### Epoch #18
![image](https://github.com/leehj825/CMPE295_mmdetection3d/assets/21224335/402270e0-6918-48d1-bcfb-7016c3ba021d)
## Troubleshoot
with Python 3.7, mmcv-full install fails for "ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS'"
```
pip install chardet
```
Failed to get GLIBC library
```
/envs/mmdet3d/lib/python3.8/ctypes/__init__.py", line 373, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /lib64/libc.so.6: version `GLIBC_2.18' not found (required by /home/001891254/miniconda3/envs/mmdet3d/lib/python3.8/site-packages/open3d/libc++abi.so.1
```
Install Open3D module
```
pip install open3d-python
```

PGD config fails with Padding related

```
  File "tools/train.py", line 135, in <module>
    main()
  File "tools/train.py", line 131, in main
    runner.train()
  File "/Users/hyejunlee/anaconda3/envs/mmdet3d/lib/python3.8/site-packages/mmengine/runner/runner.py", line 1745, in train
    model = self.train_loop.run()  # type: ignore
  File "/Users/hyejunlee/anaconda3/envs/mmdet3d/lib/python3.8/site-packages/mmengine/runner/loops.py", line 96, in run
    self.run_epoch()
  File "/Users/hyejunlee/anaconda3/envs/mmdet3d/lib/python3.8/site-packages/mmengine/runner/loops.py", line 112, in run_epoch
    self.run_iter(idx, data_batch)
  File "/Users/hyejunlee/anaconda3/envs/mmdet3d/lib/python3.8/site-packages/mmengine/runner/loops.py", line 128, in run_iter
    outputs = self.runner.model.train_step(
  File "/Users/hyejunlee/anaconda3/envs/mmdet3d/lib/python3.8/site-packages/mmengine/model/base_model/base_model.py", line 113, in train_step
    data = self.data_preprocessor(data, True)
  File "/Users/hyejunlee/anaconda3/envs/mmdet3d/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/Users/hyejunlee/mmdetection3d/mmdet3d/models/data_preprocessors/data_preprocessor.py", line 152, in forward
    return self.simple_process(data, training)
  File "/Users/hyejunlee/mmdetection3d/mmdet3d/models/data_preprocessors/data_preprocessor.py", line 170, in simple_process
    data = self.collate_data(data)
  File "/Users/hyejunlee/mmdetection3d/mmdet3d/models/data_preprocessors/data_preprocessor.py", line 264, in collate_data
    batch_imgs = stack_batch(batch_imgs, self.pad_size_divisor,
  File "/Users/hyejunlee/anaconda3/envs/mmdet3d/lib/python3.8/site-packages/mmengine/model/utils.py", line 73, in stack_batch
    F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)
```

fix by removing first 2 elements in ..//anaconda3/envs/mmdet3d/lib/python3.8/site-packages/mmengine/model/utils.py

```
    #for idx, tensor in enumerate(tensor_list):
    #    batch_tensor.append(
    #        F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    for idx, tensor in enumerate(tensor_list):
        current_pad = pad[idx].tolist()
        if tensor.dim() == 3:
            # For 3D tensor, use only the last four padding values.
            current_pad = current_pad[-4:]
        batch_tensor.append(
            F.pad(tensor, tuple(current_pad), value=pad_value))
```
for another library problem
```
pip uninstall open3d-python
pip3 install open3d
```

```
python demo/mono_det_demo.py demo/data/kitti/000008.png demo/data/kitti/000008.pkl configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d_tune.py work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d_tune/20231008_172104/epoch_7.pth --cam-type CAM2 --out-dir demo/output.png

```

![image](https://github.com/leehj825/CMPE295_mmdetection3d/assets/21224335/53d0ff07-80f9-4e02-95b7-62b8d41cb972)

