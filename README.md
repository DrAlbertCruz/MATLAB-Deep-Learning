# MATLAB Deep Learning

These functions can be used to retrain a pre-trained neural network in MATLAB 2017B. You can use a deep learning for your imaging problem, but it ussually requires a huge amount of data. By re-training an existing neural network, such as one trained on the ImageNet challenge, you can alleviate this requirement for your problem. This is called transfer learning. I found it unusual MATLAB would not provide this service as a stream-lined function. I hope that others will find this code useful.

## Practical Notes

* It might be possible to train a neural network without an Nvidia GPU, but I have never tried this and its generally not recommended.
* For VGG, GoogLenet and ResNet-50: You need a very beefy card for this. Even with a 1080ti or a K6000 (roughly 12GB VRAM) I would run out of memory. To make it work I had to set a minibatch size of 10-100 samples, which led to very long training times (epochs) to compensate. 
