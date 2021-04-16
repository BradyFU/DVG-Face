# DVG-Face: Dual Variational Generation for HFR
This repo is a [PyTorch](https://pytorch.org/) implementation of [DVG-Face: Dual Variational Generation for Heterogeneous Face Recognition](https://arxiv.org/pdf/2009.09399.pdf), which is an extension version of our previous [conference paper](https://github.com/BradyFU/DVG).


## Prerequisites
- Python 3.7.0 & PyTorch 1.5.0 & Torchvision 0.6.0
- Download LightCNN-29 ([Google Drive](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view)) pretrained on MS-Celeb-1M.
- Download Identity Sampler ([Google Drive](https://drive.google.com/file/d/1kezDpwqA4a3WGq5PfS1kDrASf3-bN4js/view?usp=sharing)) pretrained on MS-Celeb-1M.
- Put the above two models in `./pre_train`


## Train the generator
`train_generator.py`:
Fill out options of '--img_root' and '--train_list', which are the image root and training list of the heterogeneous data, respectively.
An example of the training list:
```
NIR/s2_NIR_10039_001.jpg 232
VIS/s1_VIS_00134_010.jpg 133
NIR/s1_NIR_00118_011.jpg 117
```
Here we use 'NIR' and 'VIS' in the training list to distinguish the modality of images. If your list has other distinguishing marks,
please change the corresponding marks in `./data/dataset.py` (lines 28, 38, 66, and 68).
```
python train_generator.py --gpu_ids 0
```


## Generate images from noise
`gen_samples.py`:
Fill out options of '--img_root' and '--train_list' that are the same as the above options.
```
python gen_samples.py --gpu_ids 0
```
The generated images will be saved in `./gen_images`


## Train the recognition model LightCNN-29
`train_lightcnn.py`:
Fill out options of 'num_classes', '--img_root_A', and '--train_list_A', where the last two options are the same as the above options.
```
python train_ligthcnn.py --gpu_ids 0,1
```


## Citation
If you use our code for your research, please cite the following papers:
```
@article{fu2021dvg,
  title={DVG-face: Dual variational generation for heterogeneous face recognition},
  author={Fu, Chaoyou and Wu, Xiang and Hu, Yibo and Huang, Huaibo and He, Ran},
  journal={IEEE TPAMI},
  year={2021}
}

@inproceedings{fu2019dual,
  title={Dual Variational Generation for Low-Shot Heterogeneous Face Recognition},
  author={Fu, Chaoyou and Wu, Xiang and Hu, Yibo and Huang, Huaibo and He, Ran},
  booktitle={NeurIPS},
  year={2019}
}
```








