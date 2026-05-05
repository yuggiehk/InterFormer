# InterFormer

Official implementation of the paper: **"Interaction-aware Representation Modeling with Co-occurrence Consistency for Egocentric Hand-Object Parsing"**, *ICLR 2026*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Model](#pretrained-model)
- [Training](#training)
  - [Single-GPU Training](#single-gpu-training)
  - [Multi-GPU Training](#multi-gpu-training)
- [Testing](#testing)
- [Checkpoints](#checkpoints)
- [Results](#results)

---

## Overview

This repository contains the official code for **InterFormer**, a novel framework for egocentric hand-object parsing that leverages interaction-aware representation modeling with co-occurrence consistency.

---

## Environment Setup

Create and activate the conda environment using the provided configuration file:

```bash
# Create environment from yaml file
conda env create -f mmseg.yaml

# Activate the environment
conda activate mmseg
```

> **Note**: If you don't have `mmseg.yaml`, please ensure you have installed mmsegmentation and its dependencies. Alternatively, you can install manually:
> ```bash
> pip install torch torchvision
> pip install mmcv-full
> pip install mmsegmentation
> ```

---

## Dataset Preparation

1. Download the dataset from [Google Drive](https://drive.google.com/file/d/1Pc6Ofd3q-yEwByxIqgaJfEJv4QtWtKjP/view?usp=drive_link)

2. Extract the dataset:
   ```bash
   unzip dataset.zip -d /path/to/your/dataset/folder
   ```

3. Update the dataset path in the configuration file:
   ```bash
   # Edit ./mmseg/configs/config_interformer.py
   # Change the dataset path to your extracted folder
   data_root = '/path/to/your/dataset/folder'
   ```

---

## Pretrained Model

Download the pretrained backbone model:

```bash
# Download pretrained model
wget https://drive.google.com/file/d/13Wd08wkbscxAR1TAgaNWQ2_luYC9pahs/view?usp=drive_link -O pretrained_model/backbone.pth
```

Alternatively, download manually from [this link](https://drive.google.com/file/d/13Wd08wkbscxAR1TAgaNWQ2_luYC9pahs/view?usp=drive_link) and place it in the `pretrained_model/` directory.

Then update the pretrained path in `./mmseg/configs/config_interformer.py `:
```python
pretrained = '/path/to/pretrained_model/backbone.pth'
```

---

## Training

### Single-GPU Training

Run the following command for single-GPU training:

```bash
python tools/train.py ./mmseg/configs/config_interformer.py
```

### Multi-GPU Training

For multi-GPU training, use the distributed training script:

```bash
bash tools/dist_train.sh ./mmseg/configs/config_interformer.py <NUM_GPUS>
```

Example (4 GPUs):
```bash
bash tools/dist_train.sh ./mmseg/configs/config_interformer.py py 4
```

### Training Log

We provide our training log file (`Training_Log.log`) in this repository. The best results are achieved at **iteration 93,900**.

---

## Testing

### Single-GPU Testing

To test a trained model on a single GPU:

```bash
python tools/test.py <CONFIG_FILE_PATH> <CHECKPOINT_PATH>
```

Example:
```bash
python tools/test.py config/config_interformer.py work_dirs/checkpoint.pth
```

---

## Checkpoints

We provide our trained checkpoint for reproduction:

- **Download link**: [Google Drive](https://drive.google.com/file/d/172Hi0jTTijEnbroYSUZgOfpsyue_HHxg/view?usp=drive_link)

Download and use it for testing or inference.

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{su2026interformer,
  title={Interaction-aware Representation Modeling with Co-occurrence Consistency for Egocentric Hand-Object Parsing},
  author={Su, Yuejiao and Wang, Yi and Yao, Lei and Cui, Yawen and Chau, Lap-Pui},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors directly.
