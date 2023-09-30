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
