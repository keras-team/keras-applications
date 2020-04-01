# Keras Applications

[![Build Status](https://travis-ci.org/keras-team/keras-applications.svg?branch=master)](https://travis-ci.org/keras-team/keras-applications)

Keras Applications is the `applications` module of
the Keras deep learning library.
It provides model definitions and pre-trained weights for a number
of popular archictures, such as VGG16, ResNet50, Xception, MobileNet, and more.

Read the documentation at: https://keras.io/applications/

Keras Applications may be imported directly
from an up-to-date installation of Keras:

```
from keras import applications
```

Keras Applications is compatible with Python 2.7-3.6
and is distributed under the MIT license.

## Performance

- The top-k accuracies were obtained using Keras Applications with the **TensorFlow backend** on the **2012 ILSVRC ImageNet validation set** and may slightly differ from the original ones.
  * Input: input size fed into models
  * Top-1: single center crop, top-1 accuracy
  * Top-5: single center crop, top-5 accuracy
  * Size: rounded the number of parameters when `include_top=True`
  * Stem: rounded the number of parameters when `include_top=False`

|                                                                | Input | Top-1       | Top-5       | Size   | Stem   | References                                  |
|----------------------------------------------------------------|-------|-------------|-------------|--------|--------|---------------------------------------------|
| [VGG16](keras_applications/vgg16.py)                           |  224  | 71.268      | 90.050      | 138.4M | 14.7M  | [[paper]](https://arxiv.org/abs/1409.1556) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py) |
| [VGG19](keras_applications/vgg19.py)                           |  224  | 71.256      | 89.988      | 143.7M | 20.0M  | [[paper]](https://arxiv.org/abs/1409.1556) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py) |
| [ResNet50](keras_applications/resnet50.py)                     |  224  | 74.928      | 92.060      | 25.6M  | 23.6M  | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt) |
| [ResNet101](keras_applications/resnet.py)                      |  224  | 76.420      | 92.786      | 44.7M  | 42.7M  | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-101-deploy.prototxt) |
| [ResNet152](keras_applications/resnet.py)                      |  224  | 76.604      | 93.118      | 60.4M  | 58.4M  | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-152-deploy.prototxt) |
| [ResNet50V2](keras_applications/resnet_v2.py)                  |  299  | 75.960      | 93.034      | 25.6M  | 23.6M  | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet101V2](keras_applications/resnet_v2.py)                 |  299  | 77.234      | 93.816      | 44.7M  | 42.6M  | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet152V2](keras_applications/resnet_v2.py)                 |  299  | 78.032      | 94.162      | 60.4M  | 58.3M  | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNeXt50](keras_applications/resnext.py)                     |  224  | 77.740      | 93.810      | 25.1M  | 23.0M  | [[paper]](https://arxiv.org/abs/1611.05431) [[torch]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [ResNeXt101](keras_applications/resnext.py)                    |  224  | 78.730      | 94.294      | 44.3M  | 42.3M  | [[paper]](https://arxiv.org/abs/1611.05431) [[torch]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [InceptionV3](keras_applications/inception_v3.py)              |  299  | 77.898      | 93.720      | 23.9M  | 21.8M  | [[paper]](https://arxiv.org/abs/1512.00567) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) |
| [InceptionResNetV2](keras_applications/inception_resnet_v2.py) |  299  | 80.256      | 95.252      | 55.9M  | 54.3M  | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py) |
| [Xception](keras_applications/xception.py)                     |  299  | 79.006      | 94.452      | 22.9M  | 20.9M  | [[paper]](https://arxiv.org/abs/1610.02357) |
| [MobileNet(alpha=0.25)](keras_applications/mobilenet.py)       |  224  | 51.582      | 75.792      | 0.5M   | 0.2M   | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet(alpha=0.50)](keras_applications/mobilenet.py)       |  224  | 64.292      | 85.624      | 1.3M   | 0.8M   | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet(alpha=0.75)](keras_applications/mobilenet.py)       |  224  | 68.412      | 88.242      | 2.6M   | 1.8M   | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet(alpha=1.0)](keras_applications/mobilenet.py)        |  224  | 70.424      | 89.504      | 4.3M   | 3.2M   | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNetV2(alpha=0.35)](keras_applications/mobilenet_v2.py)  |  224  | 60.086      | 82.432      | 1.7M   | 0.4M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNetV2(alpha=0.50)](keras_applications/mobilenet_v2.py)  |  224  | 65.194      | 86.062      | 2.0M   | 0.7M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNetV2(alpha=0.75)](keras_applications/mobilenet_v2.py)  |  224  | 69.532      | 89.176      | 2.7M   | 1.4M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNetV2(alpha=1.0)](keras_applications/mobilenet_v2.py)   |  224  | 71.336      | 90.142      | 3.5M   | 2.3M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNetV2(alpha=1.3)](keras_applications/mobilenet_v2.py)   |  224  | 74.680      | 92.122      | 5.4M   | 3.8M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNetV2(alpha=1.4)](keras_applications/mobilenet_v2.py)   |  224  | 75.230      | 92.422      | 6.2M   | 4.4M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNetV3(small)](keras_applications/mobilenet_v3.py)       |  224  | 68.076      | 87.800      | 2.6M   | 0.9M   | [[paper]](https://arxiv.org/abs/1905.02244) [[tf-models]](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet/mobilenet_v3.py) |
| [MobileNetV3(large)](keras_applications/mobilenet_v3.py)       |  224  | 75.556      | 92.708      | 5.5M   | 3.0M   | [[paper]](https://arxiv.org/abs/1905.02244) [[tf-models]](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet/mobilenet_v3.py) |
| [DenseNet121](keras_applications/densenet.py)                  |  224  | 74.972      | 92.258      | 8.1M   | 7.0M   | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [DenseNet169](keras_applications/densenet.py)                  |  224  | 76.176      | 93.176      | 14.3M  | 12.6M  | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [DenseNet201](keras_applications/densenet.py)                  |  224  | 77.320      | 93.620      | 20.2M  | 18.3M  | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [NASNetLarge](keras_applications/nasnet.py)                    |  331  | 82.498      | 96.004      | 93.5M  | 84.9M  | [[paper]](https://arxiv.org/abs/1707.07012) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py) |
| [NASNetMobile](keras_applications/nasnet.py)                   |  224  | 74.366      | 91.854      | 7.7M   | 4.3M   | [[paper]](https://arxiv.org/abs/1707.07012) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py) |
| [EfficientNet-B0](keras_applications/efficientnet.py)          |  224  | 77.190      | 93.492      | 5.3M   | 4.0M   | [[paper]](https://arxiv.org/abs/1905.11946) [[tf-tpu]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| [EfficientNet-B1](keras_applications/efficientnet.py)          |  240  | 79.134      | 94.448      | 7.9M   | 6.6M   | [[paper]](https://arxiv.org/abs/1905.11946) [[tf-tpu]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| [EfficientNet-B2](keras_applications/efficientnet.py)          |  260  | 80.180      | 94.946      | 9.2M   | 7.8M   | [[paper]](https://arxiv.org/abs/1905.11946) [[tf-tpu]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| [EfficientNet-B3](keras_applications/efficientnet.py)          |  300  | 81.578      | 95.676      | 12.3M  | 10.8M  | [[paper]](https://arxiv.org/abs/1905.11946) [[tf-tpu]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| [EfficientNet-B4](keras_applications/efficientnet.py)          |  380  | 82.960      | 96.260      | 19.5M  | 17.7M  | [[paper]](https://arxiv.org/abs/1905.11946) [[tf-tpu]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| [EfficientNet-B5](keras_applications/efficientnet.py)          |  456  | 83.702      | 96.710      | 30.6M  | 28.5M  | [[paper]](https://arxiv.org/abs/1905.11946) [[tf-tpu]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| [EfficientNet-B6](keras_applications/efficientnet.py)          |  528  | 84.082      | 96.898      | 43.3M  | 41.0M  | [[paper]](https://arxiv.org/abs/1905.11946) [[tf-tpu]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| [EfficientNet-B7](keras_applications/efficientnet.py)          |  600  | 84.430      | 96.840      | 66.7M  | 64.1M  | [[paper]](https://arxiv.org/abs/1905.11946) [[tf-tpu]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |


## Reference implementations from the community

### Object detection and segmentation

- [SSD](https://github.com/rykov8/ssd_keras) by @rykov8 [[paper]](https://arxiv.org/abs/1512.02325)
- [YOLOv2](https://github.com/allanzelener/YAD2K) by @allanzelener [[paper]](https://arxiv.org/abs/1612.08242)
- [YOLOv3](https://github.com/qqwweee/keras-yolo3) by @qqwweee [[paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [Mask RCNN](https://github.com/matterport/Mask_RCNN) by @matterport [[paper]](https://arxiv.org/abs/1703.06870)
- [U-Net](https://github.com/zhixuhao/unet) by @zhixuhao [[paper]](https://arxiv.org/abs/1505.04597)
- [RetinaNet](https://github.com/fizyr/keras-retinanet) by @fizyr [[paper]](https://arxiv.org/abs/1708.02002)

### Sequence learning

- [Seq2Seq](https://github.com/farizrahman4u/seq2seq) by @farizrahman4u
- [WaveNet](https://github.com/basveeling/wavenet) by @basveeling [[paper]](https://arxiv.org/abs/1609.03499)

### Reinforcement learning

- [keras-rl](https://github.com/keras-rl/keras-rl) by @keras-rl
- [RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaGo) by @Rochester-NRT [[paper]](https://doi.org/10.1038/nature16961)

### Generative adversarial networks

- [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN) by @eriklindernoren
