# Requirements

* MATLAB 2018A and above
* Nvidia GPU, refer to MATLAB's requirements for this
* Neural network toolbox
* Download support packages as needed for each neural network

# MATLAB Deep Learning

These functions can be used to retrain a pre-trained neural network in MATLAB. You can use a deep learning for your imaging problem, but it ussually requires a huge amount of data. By re-training an existing neural network, such as one trained on the ImageNet challenge, you can alleviate this requirement for your problem. This is called transfer learning. I found it unusual MATLAB would not provide this service as a stream-lined function. I hope that others will find this code useful.

## Notes

* It might be possible to train a neural network without an Nvidia GPU, but I have never tried this
