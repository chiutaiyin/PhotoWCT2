# PhotoWCT2: Compact Autoencoder for Photorealistic Style Transfer Resulting from Blockwise Training and Skip Connections of High-Frequency Residuals
Code for our WACV 2022 paper [PhotoWCT2: Compact Autoencoder for Photorealistic Style Transfer Resulting from Blockwise Training and Skip Connections of High-Frequency Residuals](https://https://arxiv.org/abs/2110.11995)

## Requirements 
- tensorflow v2.0.0 or above

## Models and files
We apply the proposed blockwise training to train two models:
1. A pre-trained VGG-19 encoder (from input layer to **conv41** layer) and a blockwisely trained decoder which can reproduce the **conv31**, **conv21**, **conv11** features and the input image. The ZCA transformations are embedded at the bottleneck and the reproduced convN1 layers in the decoder.
    - The model is in ```utils/model_conv.py``` and the associated checkpoint is in ```ckpts/ckpts-conv```.
    - A demo that uses this model to stylize example images in ```figures/``` is shown in ```conv_demo.ipynb```. The resulting stylized images are in ```results/conv```.

2. A pre-trained VGG-19 encoder (from input layer to **relu41** layer) and a blockwisely trained decoder which can reproduce the **relu31**, **relu21**, **relu11** features and the input image. The ZCA transformations are embedded at the bottleneck and the reproduced reluN1 layers in the decoder.
    - The model is in ```utils/model_relu.py``` and the associated checkpoint is in ```ckpts/ckpts-relu```.
    - A demo that uses this model to stylize example images in ```figures/``` is shown in ```relu_demo.ipynb```. The resulting stylized images are in ```results/relu```.

Stylization with both models requires guided filtering in ```utils/photo_gif.py``` as the post-processing. The file is adapted from the one used in the [PhotoWCT code](https://github.com/NVIDIA/FastPhotoStyle).

The performance of these two models is similar. Please choose whichever you would like to use.

## Citation
If you find this repo useful, please cite our paper **PhotoWCT2: Compact Autoencoder for Photorealistic Style Transfer Resulting from Blockwise Training and Skip Connections of High-Frequency Residuals** published in WACV 2022.
