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
        <li><a href="#Data-Collection">Data Collection</a></li>
        <li><a href="#Video-Sampling-Rate">Video Sampling Rate</a></li>
      </ul>
    </li>
    <li>
      <a href="#About-the-Model">About the Model</a>
      <ul>
        <li><a href="#Architecture">Architecture</a></li>
        <li><a href="#Data-Augmentation-Schemes">Data Augmentation Schemes</a></li>
        <li><a href="#Training">Training</a></li>
        <li><a href="#Ablation-Study">Ablation Study</a></li>
      </ul>
    </li>
    <li><a href="#Reference">Reference</a></li>
  </ol>
</details>


---
## About the dataset
### Data Collection
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

### Data Augmentation Schemes
`data.py supports the following operations currently:`
- Gaussian Noise with 'mean=0' and `std=1e-4`.
- Selecting video clips (>20 frames) randomly.


### Training
- CPU is enough for training the official data.
- Running `python train.py` in command line.


### Ablation Study
- Vanilla: training with no tricks.
- DA: training with data augmentation schemes.

| Model | Method | Units of `nn.Linear()` | Model Size | Acc|
| :---: | :---: | :---: | :---: | :---: |
| ActionNet | Vanilla | 512 | 28.95M | 0.9250 |
| ActionNet | DA | 512 |  28.95M | 0.9875 |
| ActionNet | Vaniila | 1280 | 50.61M | 0.9750 |
| ActionNet | DA | 1280 |  50.61M | 1.000 |
| ActionNet | Vanilla | 2048 | 92.29M | 0.9875 |
| ActionNet | DA | 2048 | 92.29M | 1.000 |

---
## Reference
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [DSNet: A Flexible Detect-to-Summarize Network for Video Summarization](https://github.com/li-plus/DSNet) (TIP, 2020)
- [ML-Decoder: Scalable and Versatile Classification Head](https://arxiv.org/abs/2111.12933) (arXiv, 2021)
