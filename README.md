# DVG-Face: Dual Variational Generation for HFR
This repo is a [PyTorch](https://pytorch.org/) implementation of [DVG-Face: Dual Variational Generation for Heterogeneous Face Recognition](https://arxiv.org/pdf/2009.09399.pdf), which is an extension version of our previous [conference paper](https://github.com/BradyFU/DVG).


<p align="center">  
<img src="image/framework.png">  
</p> 


## Prerequisites
- Python 3.7.0
- Pytorch 1.5.0 & torchvision 0.6.0

## Train the generator
- Download LightCNN-29 ([Google Drive](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view)) pretrained on MS-Celeb-1M.
- Download Identity Sampler ([Google Drive](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view)) pretrained on MS-Celeb-1M.
- Put the above two models in `./pre_train`






