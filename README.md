# CMPE295_Project

The project is based on the MMDetection3D repository, https://github.com/open-mmlab/mmdetection3d

## Installation
Mostly following the instructions in MMDetection3D GitHub, below steps work in SJSU HPC. 

### CUDA and Pytorch
```
conda install -c conda-forge cudatoolkit=11.7.0
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```
### MMDetection3D
```
pip install openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
mim install "mmdet3d>=1.1.0"
```
### In SJSU HPC, enable GPU node and confirm CUDA is enabled
```
srun -p gpu --gres=gpu -n 1 -N 1 -c 4 --pty /bin/bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Models and Code Bases
- **MMDetection3D**: https://github.com/open-mmlab/mmdetection3d

- **Model**
   - PGD (FCOS3D++) https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pgd
   - FCOS3D https://github.com/open-mmlab/mmdetection3d/tree/main/configs/fcos3d
   - SMOKE https://github.com/open-mmlab/mmdetection3d/tree/main/configs/smoke

- **Dataset**
    - Waymo
    - Kitti

## Dataset Preparation
- Waymo Dataset: https://waymo.com/open/download/
- Kitti Dataset: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

### Waymo 1.4.1 Dataset
From Waymo Open database, 499 tfrecord files are used for training, and 202 tfrecords are used for validation.  Due to the size, the entire dataset is located in SJSU HPC server (/scratch/cmpe295-wu/hj/waymo_data/), and it is not available for download.   For quick trial and validation, single tfrecord data for training and validation is available for download. 

Download **waymo_single.zip** (5GB) from below link and unzip.
- https://drive.google.com/file/d/1Cwsr4xLuDIuomnQqopG1xYvVkS8GpQju/view?usp=drive_link

To simulate federated learnings, datasets are separated into 4 groups (FedAvg4).  Then each sub dataset is split half for 8 groups (FedAvg8)
```
├── waymo_1_4_1
│   ├── all (Centralized Training)
│   ├── part_1 (FedAvg4)
│   ├── part_2 (FedAvg4)
│   ├── part_3 (FedAvg4)
│   ├── part_4 (FedAvg4)
│   ├── n8
│   │   ├── part_1 (FedAvg8)
│   │   ├── part_2 (FedAvg8)
│   │   ├── part_3 (FedAvg8)
│   │   ├── part_4 (FedAvg8)
│   │   ├── part_5 (FedAvg8)
│   │   ├── part_6 (FedAvg8)
│   │   ├── part_7 (FedAvg8)
│   │   ├── part_8 (FedAvg8)
```

## Training
For federated learning process, four separate datasets are used four clients, with a global model updated in every 5 epochs. Each client trains locally on its dataset for five epochs. The local model parameters are aggregated in the global model. The easiest method of aggregation is simply averaging, known as FedAvg. This updated global model then serves as the starting point for the next training round on each client. This cycle of local training, averaging, and updating continues for multiple rounds.

### PGD config
#### Train dataset #1: configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_1.py

```
_base_ = [
    '../_base_/datasets/waymoD5-fov-mono3d-3class_1.py',
    '../_base_/models/pgd.py', '../_base_/schedules/mmdet-schedule-1x.py',
    '../_base_/default_runtime.py'
]

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
:
:
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
```
The training script is provided under tools folder.
```
> python tools/train.py configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_1.py
```
### Federated Averaging
After 5 epochs of training in each client, the local checkpoints are aggregated with FedAvg.
```
> python fedavg.py configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_1_epoch_5.py
```
<img width="855" alt="image" src="https://github.com/leehj825/CMPE295_mmdetection3d/assets/21224335/5601470b-0814-4ce8-939f-9fb912d2f9c4">

### Training Time

- Training dataset: 499 tfrecord files
- Validtion dataset: 202 tfrecord files

|               | training (5 ep) | validation     | Total Time (7 iterations) | Training Only |
|---------------|----------------:|---------------:|--------------------------:|--------------:|
| FedAvg8       |         3 hours |        4 hours |                 2.04 days |     0.88 days |
| FedAvg4       |         6 hours |        4 hours |                 2.92 days |     1.75 days |
| Centralized   |        34 hours |        4 hours |                11.08 days |     9.92 days |


### LET-mAP Metric Evaluation

When utilizing federated learning with averaging, the approach demonstrates effectiveness over centralized training, using the same dataset comprising 202 tfrecord files for evaluation. Initially, both FedAvg4 and FedAvg8 models started with lower performance metrics compared to the centralized training model. However, by the 20th epoch, FedAvg8 began to surpass the centralized model's performance, indicating a delayed but eventual benefit to the federated learning process. On the other hand, FedAvg4 not only improved over the centralized model earlier, by the 15th epoch, but it also consistently outperformed both FedAvg8 and centralized training across subsequent epochs. 

<img width="600" alt="image" src="https://github.com/leehj825/CMPE295_mmdetection3d/assets/21224335/74bca831-737b-4c6f-9e64-c63ed548a35b">

### Inference

<img width="600" alt="image" src="https://github.com/leehj825/CMPE295_mmdetection3d/assets/21224335/709b74eb-7a2a-4b9e-af1c-314093fa5391)">


## Appendix

### Create Kitti Format Annotation
Refer: https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html

Download Kitti dataset to below folder structure
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

#### Script to create kitti pickle annotation files
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
### Convert Waymo tfrecord to kitti format
Download tfrecords (1.4.1) from here https://waymo.com/open/download/

#### Waymo format structure
```
│   ├── waymo
│   │   ├── waymo_format (each subfolder containing tfrecord files)
│   │   │   ├── testing
│   │   │   ├── testing_3d_camera_only_detection
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── kitti_format (output files from script)
```
#### Convert to kitti format
```
> python tools/create_data.py waymo --root-path data/waymo --out-dir data/waymo --extra-tag waymo

:
:

tfrecord_pathnames(0) data/waymo/waymo_format/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
tfrecord_pathnames(1) data/waymo/waymo_format/training/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord
2023-10-12 06:59:41.883071: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2023-10-12 06:59:41.885273: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
tfrecord_pathnames(3) data/waymo/waymo_format/training/segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord
2023-10-12 06:59:41.910530: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
tfrecord_pathnames(4) data/waymo/waymo_format/training/segment-10072140764565668044_4060_000_4080_000_with_camera_labels.tfrecord
tfrecord_pathnames(5) data/waymo/waymo_format/training/segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord
[>>                                                ] 1/21, 0.0 task/s, elapsed: 562s, ETA: 11240stfrecord_pathnames(6) data/waymo/waymo_format/training/segment-10075870402459732738_1060_000_1080_000_with_camera_labels.tfrecord
[>>>>>>>>>                                         ] 4/21, 0.0 task/s, elapsed: 622s, ETA:  2644stfrecord_pathnames(7) data/waymo/waymo_format/training/segment-10082223140073588526_6140_000_6160_000_with_camera_labels.tfrecord
tfrecord_pathnames(8) data/waymo/waymo_format/training/segment-10094743350625019937_3420_000_3440_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>                                    ] 6/21, 0.0 task/s, elapsed: 1107s, ETA:  2767stfrecord_pathnames(9) data/waymo/waymo_format/training/segment-10096619443888687526_2820_000_2840_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>                                  ] 7/21, 0.0 task/s, elapsed: 1124s, ETA:  2248stfrecord_pathnames(10) data/waymo/waymo_format/training/segment-10107710434105775874_760_000_780_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>>>>                               ] 8/21, 0.0 task/s, elapsed: 1184s, ETA:  1924stfrecord_pathnames(11) data/waymo/waymo_format/training/segment-10153695247769592104_787_000_807_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>>>>>>                             ] 9/21, 0.0 task/s, elapsed: 1636s, ETA:  2182stfrecord_pathnames(12) data/waymo/waymo_format/training/segment-1022527355599519580_4866_960_4886_960_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>>>>>>>>                           ] 10/21, 0.0 task/s, elapsed: 1669s, ETA:  1836stfrecord_pathnames(13) data/waymo/waymo_format/training/segment-10231929575853664160_1160_000_1180_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>>>>>>>>>>>                        ] 11/21, 0.0 task/s, elapsed: 1677s, ETA:  1524stfrecord_pathnames(14) data/waymo/waymo_format/training/segment-10235335145367115211_5420_000_5440_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>                      ] 12/21, 0.0 task/s, elapsed: 1725s, ETA:  1294stfrecord_pathnames(15) data/waymo/waymo_format/training/segment-10241508783381919015_2889_360_2909_360_with_camera_labels.tfrecord
```
#### Waymo dataset Output in Kitti Format

```
│   ├── kitti_format
│   │   ├── ImageSets
│   │   ├── testing_3d_camera_only_detection
│   │   │   ├── image_0
│   │   │   ├── image_1
│   │   │   ├── image_2
│   │   │   ├── image_3
│   │   │   ├── [others that are not used for this project]
│   │   ├── training
│   │   │   ├── image_0
│   │   │   ├── image_1
│   │   │   ├── image_2
│   │   │   ├── image_3
│   │   │   ├── label_0
│   │   │   ├── label_1
│   │   │   ├── label_2
│   │   │   ├── label_3
│   │   │   ├── [others that are not used for this project]
│   │   ├── waymo_infos_test.pkl
│   │   ├── waymo_infos_train.pkl
│   │   ├── waymo_infos_trainval.pkl
│   │   ├── waymo_infos_val.pkl
```
Object label example in Waymo dataset
```
+------------+--------+
| category   | number |
+------------+--------+
| Car        | 33375  |
| Pedestrian | 8556   |
| Cyclist    | 294    |
+------------+--------+
```

## KITTI Dataset
### Kitti 2017 Dataset without data processing
Download **kitti_data.zip** from below link and unzip.
- https://drive.google.com/file/d/1r_kvJ2zTgeu6X5QUtSaYa9PIpauz0Hrh/view?usp=sharing

To simulate federated learnings, datasets are separated into 4 groups
```
│   ├── kitti_1
│   ├── kitti_2
│   ├── kitti_3
│   ├── kitti_4
```

For creating new Kitti dataset or converting Waymo dataset, please refer to Appendix below.
### Loss value from kitti training
<img width="922" alt="image" src="https://github.com/leehj825/CMPE295_mmdetection3d/assets/21224335/4607003b-126a-49ba-b932-1a46a888ead3">

## Inference
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
