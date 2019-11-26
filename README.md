# G-ResNeXt_GroupNet
This repository reproduces the results of the following paper:
[**Differentiable Learning-to-Group Channels via Groupable Convolutional Neural Networks**](https://arxiv.org/abs/1908.05867v1)
re-implement of Group ConvNet, also be called as G-ResNext. It's from the paper, reproduction of the paper "Differentiable Learning-to-Group Channels via Groupable Convolutional Neural Networks".
The architecture is the same as G-ResNeXt in table 1 of the paper. I just re-implemented the GroupNet bu using dynamic grouping convolution (DGConv) operation.


# Guideline for train the G-ResNeXt-50, 101 on ImageNet.
* just change the imagenet data path, change the GPU ID for fast reproduceing of the GroupConvNet.
* PyTorch 0.4.0+, 1.0 is all ok.
* Pretrained weights can be downloaded from BaiduYunPan (Baidu drive).

