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

- The top-k errors were obtained using Keras Applications with the **TensorFlow backend** on the **2012 ILSVRC ImageNet validation set** and may slightly differ from the original ones.
The input size used was 224x224 for all models except NASNetLarge (331x331), InceptionV3 (299x299), InceptionResNetV2 (299x299), and Xception (299x299).
  * Top-1: single center crop, top-1 error
  * Top-5: single center crop, top-5 error
  * 10-5: ten crops (1 center + 4 corners and those mirrored ones), top-5 error
  * Size: rounded the number of parameters when `include_top=True`
  * Stem: rounded the number of parameters when `include_top=False`

|                                                                | Top-1       | Top-5       | 10-5        | Size   | Stem   | References                                  |
|----------------------------------------------------------------|-------------|-------------|-------------|--------|--------|---------------------------------------------|
| [VGG16](keras_applications/vgg16.py)                           | 28.732      | 9.950       | 8.834       | 138.4M | 14.7M  | [[paper]](https://arxiv.org/abs/1409.1556) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py) |
| [VGG19](keras_applications/vgg19.py)                           | 28.744      | 10.012      | 8.774       | 143.7M | 20.0M  | [[paper]](https://arxiv.org/abs/1409.1556) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py) |
| [ResNet50](keras_applications/resnet50.py)                     | 25.072      | 7.940       | 6.828       | 25.6M  | 23.6M  | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt) |
| [ResNet101](keras_applications/resnet.py)                      | 23.580      | 7.214       | 6.092       | 44.7M  | 42.7M  | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-101-deploy.prototxt) |
| [ResNet152](keras_applications/resnet.py)                      | 23.396      | 6.882       | 5.908       | 60.4M  | 58.4M  | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-152-deploy.prototxt) |
| [ResNet50V2](keras_applications/resnet_v2.py)                  | 24.040      | 6.966       | 5.896       | 25.6M  | 23.6M  | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet101V2](keras_applications/resnet_v2.py)                 | 22.766      | 6.184       | 5.158       | 44.7M  | 42.6M  | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet152V2](keras_applications/resnet_v2.py)                 | 21.968      | 5.838       | 4.900       | 60.4M  | 58.3M  | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNeXt50](keras_applications/resnext.py)                     | 22.260      | 6.190       | 5.410       | 25.1M  | 23.0M  | [[paper]](https://arxiv.org/abs/1611.05431) [[torch]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [ResNeXt101](keras_applications/resnext.py)                    | 21.270      | 5.706       | 4.842       | 44.3M  | 42.3M  | [[paper]](https://arxiv.org/abs/1611.05431) [[torch]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [InceptionV3](keras_applications/inception_v3.py)              | 22.102      | 6.280       | 5.038       | 23.9M  | 21.8M  | [[paper]](https://arxiv.org/abs/1512.00567) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) |
| [InceptionResNetV2](keras_applications/inception_resnet_v2.py) | 19.744      | 4.748       | 3.962       | 55.9M  | 54.3M  | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py) |
| [Xception](keras_applications/xception.py)                     | 20.994      | 5.548       | 4.738       | 22.9M  | 20.9M  | [[paper]](https://arxiv.org/abs/1610.02357) |
| [MobileNet(alpha=0.25)](keras_applications/mobilenet.py)       | 48.418      | 24.208      | 21.196      | 0.5M   | 0.2M   | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet(alpha=0.50)](keras_applications/mobilenet.py)       | 35.708      | 14.376      | 12.180      | 1.3M   | 0.8M   | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet(alpha=0.75)](keras_applications/mobilenet.py)       | 31.588      | 11.758      | 9.878       | 2.6M   | 1.8M   | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet(alpha=1.0)](keras_applications/mobilenet.py)        | 29.576      | 10.496      | 8.774       | 4.3M   | 3.2M   | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNetV2(alpha=0.35)](keras_applications/mobilenet_v2.py)  | 39.914      | 17.568      | 15.422      | 1.7M   | 0.4M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNetV2(alpha=0.50)](keras_applications/mobilenet_v2.py)  | 34.806      | 13.938      | 11.976      | 2.0M   | 0.7M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNetV2(alpha=0.75)](keras_applications/mobilenet_v2.py)  | 30.468      | 10.824      | 9.188       | 2.7M   | 1.4M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNetV2(alpha=1.0)](keras_applications/mobilenet_v2.py)   | 28.664      | 9.858       | 8.322       | 3.5M   | 2.3M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNetV2(alpha=1.3)](keras_applications/mobilenet_v2.py)   | 25.320      | 7.878       | 6.728       | 5.4M   | 3.8M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNetV2(alpha=1.4)](keras_applications/mobilenet_v2.py)   | 24.770      | 7.578       | 6.518       | 6.2M   | 4.4M   | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [DenseNet121](keras_applications/densenet.py)                  | 25.028      | 7.742       | 6.522       | 8.1M   | 7.0M   | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [DenseNet169](keras_applications/densenet.py)                  | 23.824      | 6.824       | 5.860       | 14.3M  | 12.6M  | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [DenseNet201](keras_applications/densenet.py)                  | 22.680      | 6.380       | 5.466       | 20.2M  | 18.3M  | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [NASNetLarge](keras_applications/nasnet.py)                    | 17.502      | 3.996       | 3.412       | 93.5M  | 84.9M  | [[paper]](https://arxiv.org/abs/1707.07012) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py) |
| [NASNetMobile](keras_applications/nasnet.py)                   | 25.634      | 8.146       | 6.758       | 7.7M   | 4.3M   | [[paper]](https://arxiv.org/abs/1707.07012) [[tf-models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py) |


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
