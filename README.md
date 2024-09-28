# FusionUNet
This repository contains a PyTorch implementation of U-Net using High/Low Level Sensorfusion.

## Requirement

```shell script
pip install -U -r requirements.txt
```

### Training and Evaluation
For Training on distortion blurring of level 1 with probability 0.7 on both data sources:
```
python3 train_eval_routine.py \
    --disturb_type_training 'blur blur' \
    --disturb_prop_training '0.7 0' \
    --disturb_level_training '1 1' \
    --saved_fn 'performance/' \
    --data_root './data/'
```
