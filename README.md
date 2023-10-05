# CMPE295_mmdetection3d

Start from https://github.com/open-mmlab/mmdetection3d

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

fix by removing first 2 elements

```    #for idx, tensor in enumerate(tensor_list):
    #    print(pad[idx])
    #    print(pad[idx].tolist())
    #    print(tuple(pad[idx].tolist()))
    #    #print(tensor)
    #    print(tensor.shape)
    #    #tensor = tensor.unsqueeze(0)
    #    print(tensor.shape)
    #    print(batch_tensor)
    #    batch_tensor.append(
    #        F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    for idx, tensor in enumerate(tensor_list):
        current_pad = pad[idx].tolist()
        if tensor.dim() == 3:
            # For 3D tensor, use only the last four padding values.
            current_pad = current_pad[-4:]
        batch_tensor.append(
            F.pad(tensor, tuple(current_pad), value=pad_value))
    return torch.stack(batch_tensor)
```
