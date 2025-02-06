We used the neural signal dataset (ECoG) for handwritten character recognition from the paper [1]. In [1], the authors acquired ECoG signals from a patient who provided his neural signals while writing 31 characters and sentences in 10 separate sessions, with each character having 117 samples. In [1], the authors proposed a complex CuDNNGRU model for detecting the handwritten character from the neural data. We, on the other hand, wanted to perform inferencing on a portable platform, and hence, could not use any complex model. Because of this reason, we represented the raw neural signals data as images and used MLP, Lenet5, MobileNetV2, ResNet18 and EfficientNet_B0, Alexnet and Resnet50 models on the raw dataset. However, the raw dataset exhibited overfitting on the Machine Learning models. Consequently, we used random noise injection and time-shifting-based data augmentation on the raw dataset, which makes the data 3 times larger than the raw dataset and prevented the overfitting challenges.

The dataaugmentation.py file is used for data augmentation.

Raw and augmented dataset link: https://uflorida-my.sharepoint.com/:f:/g/personal/ovishake_sen_ufl_edu/EkPqQUHmq6pBjV1rUkwFfqUBpKY6shXNXoQgDGaDFFBLCA?e=9YfcUg

The models are trained using the (mlp,lenet5,mobilenetv2,resnet18,efficientnet-lite).py files and the real-time inferencing on Nvidia Jetson is done using without_autocorrect_(mlp,lenet5,mobilenetv2,resnet18,efficientnet-lite).py files
Reference:
1. Willett et al., “High-performance brain-to-text communication via handwriting,” Nature, vol. 593, no. 7858, pp. 249–254, 2021.
