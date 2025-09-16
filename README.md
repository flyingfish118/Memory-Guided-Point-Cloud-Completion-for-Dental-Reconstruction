## Usage

### Requirements

- PyTorch >= 1.7.0
- python >= 3.7
- CUDA >= 9.0
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX

```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

```
# Chamfer Distance
bash install.sh
```
The solution for a common bug in chamfer distance installation can be found in Issue [#6](https://github.com/yuxumin/PoinTr/issues/6)
```
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Note: If you still get `ModuleNotFoundError: No module named 'gridding'` or something similar then run these steps

```
    1. cd into extensions/Module (eg extensions/gridding)
    2. run `python setup.py install`
```

That will fix the `ModuleNotFoundError`.

### Evaluation

To evaluate a pre-trained PoinTr model on the Three Dataset with single GPU, run:

```
bash ./scripts/test.sh <GPU_IDS>  \
    --ckpts <path> \
    --config <config> \
    --exp_name <name> \
```

####  Some examples:
```
bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/ckpt-best.pth \
    --config ./cfgs/T3DS_models/Teeth.yaml \
    --exp_name example
```

### Training

```
# Use DistributedDataParallel (DDP)
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]

# or just use DataParallel (DP)
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
```
####  Some examples:
Train a PoinTr model on PCN benchmark with 2 gpus:
```
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config ./cfgs/T3DS_models/Teeth.yaml \
    --exp_name example
```

Train a PoinTr model with a single GPU:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config ./cfgs/T3DS_models/Teeth.yaml \
    --exp_name example
```

## License
MIT License

## Acknowledgements

Our code is inspired by [PoinTr](https://github.com/yuxumin/PoinTr).

