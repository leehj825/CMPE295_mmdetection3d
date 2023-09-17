# CMPE295_mmdetection3d

Start from https://github.com/open-mmlab/mmdetection3d

for error from "mim install mmengine"

.../anaconda3/envs/openmmlab/lib/python3.8/site-packages/charset_normalizer/api.py", line 10, in <module>
    from charset_normalizer.md import mess_ratio
  File "charset_normalizer/md.py", line 5, in <module>
ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant' (/Users/hyejunlee/anaconda3/envs/openmmlab/lib/python3.8/site-packages/charset_normalizer/constant.py)

run

pip install chardet
