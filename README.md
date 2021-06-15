# KDNet

## Introduction

Object detection is a challenging computer vision task with numerous real-world applications. In recent years, the opinion of modeling relationships between objects is helpful for object detection was verified and realized in deep learning. That being said, most of the approaches of modeling object relations are limited to follow the anchor-based algorithms. It cannot be directly migrated in the anchor-free framework. The reason is that this type of algorithm eliminates the use of anchors, predicts heatmaps to represent the locations of key-points of different object categories, without considering the relationship between key-points. Therefore, to better fuse the information between the heatmap channels, it is important to model the visual relationship between key-points. In this paper, we present a knowledge driven network (KDNet)—a new architecture that can aggregate and model key-points relation to augment object features for detection. Specifically, it processes a set of key-points simultaneously through interaction between their local feature and geometry feature thereby allowing modeling of their relationship. Finally, the updated heatmaps are used to get the corner of the objects and determine the position of the objects. The experimental results conducted on the RIDER dataset confirm the effectiveness of the proposed KDNet, which significantly outperforms other state-of-the-art object detection methods.

This repository is based on [CornerNet-Lite]([princeton-vl/CornerNet-Lite (github.com)](https://github.com/princeton-vl/CornerNet-Lite)), we released our code, and the rest will follow.

## Install

Create environment reference [CornerNet-Lite]([princeton-vl/CornerNet-Lite (github.com)](https://github.com/princeton-vl/CornerNet-Lite))

## Train

To train a model:

```cmd
sh train_start.sh
```

We write the training code as a shell script, which can generate the training log for debug.

## Evaluation

You can use your own training model, or use my open source model, which is available on BaiduYun.

Link：https://pan.baidu.com/s/1BqhwUFZtQxAjK1maAKgbXA 
Code：wcg9 

To evaluation a model:

```
sh test_start.sh
```

We write the evaluation code as a shell script, which can generate the evaluation log for debug.

## Visualization

After downloading the models, you also be able to use the detectors on your own images. We provide a demo script `demo.py` to test if the repo is installed correctly.

```
python demo.py
```

