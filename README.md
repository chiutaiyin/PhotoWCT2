# PhotoWCT2: Compact Autoencoder for Photorealistic Style Transfer Resulting from Blockwise Training and Skip Connections of High-Frequency Residuals
Code for our WACV 2022 paper [PhotoWCT2: Compact Autoencoder for Photorealistic Style Transfer Resulting from Blockwise Training and Skip Connections of High-Frequency Residuals](https://openaccess.thecvf.com/content/WACV2022/html/Chiu_PhotoWCT2_Compact_Autoencoder_for_Photorealistic_Style_Transfer_Resulting_From_Blockwise_WACV_2022_paper.html)
- Our PhotoWCT2 transfers stronger effects than [WCT2](https://github.com/clovaai/WCT2) and comparable effects to those transferred by [PhotoWCT](https://github.com/NVIDIA/FastPhotoStyle).
![alt text](https://github.com/chiutaiyin/PhotoWCT2/blob/master/banner/banner.jpg)
- Our PhotoWCT2 preserves better high-frequency details than PhotoWCT (three examples are shown below).
![alt text](https://github.com/chiutaiyin/PhotoWCT2/blob/master/banner/phwct2_vs_phwct.jpg)


## Models and files
We apply the proposed blockwise training to train two models:
1. A pre-trained VGG-19 encoder (from input layer to **conv41** layer; fixed during training) and a blockwisely trained decoder which can reproduce the **conv31**, **conv21**, **conv11** features and the input image. The ZCA transformations are embedded at the bottleneck and the reproduced convN1 layers in the decoder.
    - The model is in ```utils/model_conv.py``` and the associated checkpoint is in ```ckpts/ckpts-conv```.
    - A demo that uses this model to stylize example images in ```figures/``` is shown in ```conv_demo.ipynb```. The resulting stylized images are in ```results/conv```.

2. A pre-trained VGG-19 encoder (from input layer to **relu41** layer; fixed during training) and a blockwisely trained decoder which can reproduce the **relu31**, **relu21**, **relu11** features and the input image. The ZCA transformations are embedded at the bottleneck and the reproduced reluN1 layers in the decoder.
    - The model is in ```utils/model_relu.py``` and the associated checkpoint is in ```ckpts/ckpts-relu```.
    - A demo that uses this model to stylize example images in ```figures/``` is shown in ```relu_demo.ipynb```. The resulting stylized images are in ```results/relu```.

Stylization with both models requires guided filtering in ```utils/photo_gif.py``` as the post-processing. The file is adapted from the one used in the [PhotoWCT code](https://github.com/NVIDIA/FastPhotoStyle).

The performance of these two models is similar. Please choose whichever you would like to use.

## Training
```train.py``` is the training code for our PhotoWCT2 model. The usage is provided in the file.

## Requirements 
- tensorflow v2.0.0 or above (we developed the models with tf-v2.4.1 and we also tested them in tf-v2.0.0)

## Citation
If you find this repo useful, please cite our paper **PhotoWCT2: Compact Autoencoder for Photorealistic Style Transfer Resulting from Blockwise Training and Skip Connections of High-Frequency Residuals** published in WACV 2022.
