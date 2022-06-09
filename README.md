# HUAWEI CSIG 2022: Action Recognition Challenge

[![Rank 2](https://img.shields.io/badge/DS%20ActionNet-Solution%20of%20CSIG%20Action%20Recognition%20Challenge-brightgreen.svg?style=flat-square)](https://github.com/ForeverPs/huawei_csig_action_recognition)

<img src="https://github.com/ForeverPs/huawei_csig_action_recognition/blob/main/image/csig.png" width="600px"/>

---

[Homepage](https://competition.huaweicloud.com/information/1000041695/circumstance) |
[Checkpoints](https://drive.google.com/drive/u/0/folders/1-Pn1Vhltks00zwnLnMQ6FFJJNO0OXGL4)


Official PyTorch Implementation

> Sen Pei, Jiaxi Sun
> <br/> Institute of Automation, Chinese Academy of Sciences
---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#About-the-dataset">About the dataset</a>
      <ul>
        <li><a href="#Data-Collection">Data-collection</a></li>
        <li><a href="#Video-Sampling-Rate">Video Sampling Rate</a></li>
      </ul>
    </li>
    <li>
      <a href="#About-the-Model">About the Model</a>
      <ul>
        <li><a href="#Architecture">Architecture</a></li>
        <li><a href="#Ablation-Study">Ablation Study</a></li>
        <li><a href="#Data-Augmentation-Schemes">Data Augmentation Schemes</a></li>
        <li><a href="#Training">Training</a></li>
      </ul>
    </li>
    <li><a href="#Reference">Reference</a></li>
  </ol>
</details>


---
## About the dataset
### Data collection
- We collect auxiliary videos belong to 6 persons within 10 categories.
- The collected data is not used in training currently.

### Video Sampling Rate
- Officially, 30 fps.
- The duration of each video is 20 seconds approximately.
---

## About the Model
### Architecture
- A single `nn.Linear()` layer to model the input angle parameters.
- Self attention scheme is used fro temporal feature fusion.
- Shortcut is used for improving the classification performance.
- [ML Decoder](https://github.com/Alibaba-MIIL/ML_Decoder) is the classification head.

<img src="https://github.com/ForeverPs/huawei_csig_action_recognition/blob/main/image/model.jpg" width="600px"/>

### Ablation Study

| Model | Method | Top 1 Acc | FGSM Linf=8/255 | PGD L1=1600 | PGD L2=8.0 | PGD Linf=8/255 | C&W L2=8.0 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ActionNet | Vanilla | 77.89 | 31.77 | 0.01 | 0.01 | 0.00 | 0.21 |
| ResNet-50 | GCM | 78.57 | 95.18 | 94.82 | 94.41 | 97.38 | 95.11 |
| WideResNet-50 | Vanilla | 78.21 | 20.88 | 0.36 | 0.61 | 0.50 | 0.21 |
| WideResNet-50 | GCM | 78.08 |  96.06 | 94.46 | 94.51 | 97.69 | 95.66 |
| DenseNet-121 | Vanilla | 74.86 |  16.82 | 0.04 | 0.05 | 0.06 | 0.12 |
| DenseNet-121 | GCM | 74.71 | 94.98 | 94.31 | 94.08 | 97.16 | 95.49 |
| EfficientNet-B4 | Vanilla | 71.52 | 1.23 | 0.36 | 0.28 | 0.20 | 1.88 |
| EfficientNet-B4 | GCM | 71.76 | 94.68 | 89.95 | 90.87 | 97.97 | 93.07 |
| ViT-B/16 | Vanilla | 79.46 | 15.86 | 0.00 | 0.00 | 0.00 | 0.90 |
| ViT-B/16 | GCM | 79.47 | 92.24 | 94.94 | 95.07 | 98.24 | 93.31 |
| Swin-Transformer-S | Vanilla | 82.93 | 16.93 | 0.20 | 0.00 | 0.00 | 0.76 |
| Swin-Transformer-S | GCM | 82.79 | 94.38 | 90.71 | 91.04 | 98.77 | 92.31 |

### Data Augmentation Schemes
`data_aug.py supports the following operations currently:`
- PepperSaltNoise
- ColorPointNoise
- GaussianNoise
- Mosaic in black / gray / white / color
- RGBShuffle / ColorJitter
- Rotate
- HorizontalFlip / VerticalFlip


### Training
- `python -m torch.distributed.launch --nproc_per_node=5 train.py  --batch_size 64 --n_gpus=5`
- If you have more GPUs, you can modify the `nproc_per_node` and `n_gpus` to utilize them.

---
## Reference
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (NeurIPS, 2014)
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (CVPR, 2018)
- [Mitigating adversarial effects through randomization](https://arxiv.org/abs/1711.01991) (ICLR, 2018)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899v2.pdf) (ICCV, 2019)
- [A ConvNet for the 2020s](https://github.com/facebookresearch/ConvNeXt) (CVPR, 2022)
