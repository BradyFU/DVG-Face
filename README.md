# DVG-Face: Dual Variational Generation for HFR
This repo is a [PyTorch](https://pytorch.org/) implementation of [DVG-Face: Dual Variational Generation for Heterogeneous Face Recognition](https://arxiv.org/pdf/2009.09399.pdf), which is an extension version of our previous [conference paper](https://github.com/BradyFU/DVG). Compared with the previous one, this version has more powerful performances.


## Our Heterogeneous Face Recognition Works

✨ [**DVG-Face: Dual Variational Generation for Heterogeneous Face Recognition**](https://arxiv.org/pdf/2009.09399.pdf)
- Chaoyou Fu, Xiang Wu, Yibo Hu, Huaibo Huang, and Ran He. **IEEE TPAMI 2021**

✨ [**Dual Variational Generation for Low Shot Heterogeneous Face Recognition**](https://proceedings.neurips.cc/paper/2019/file/b5a1fc2085986034e448d2ccc5bb9703-Paper.pdf)
- Chaoyou Fu, Xiang Wu, Yibo Hu, Huaibo Huang, and Ran He. **NeurIPS 2019**

✨ [**Towards Lightweight Pixel-Wise Hallucination for Heterogeneous Face Recognition**](https://ieeexplore.ieee.org/abstract/document/9971748)
- Chaoyou Fu, Xiaoqiang Zhou, Weizan He, and Ran He. **IEEE TPAMI 2022**

✨ [**Cross-Spectral Face Hallucination via Disentangling Independent Factors**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Duan_Cross-Spectral_Face_Hallucination_via_Disentangling_Independent_Factors_CVPR_2020_paper.pdf)
- Boyan Duan, Chaoyou Fu, Yi Li, Xingguang Song, and Ran He. **CVPR 2020**


## Prerequisites
- Python 3.7.0 & PyTorch 1.5.0 & Torchvision 0.6.0
- Download LightCNN-29 [[Google Drive](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view)] pretrained on MS-Celeb-1M.
- Download Identity Sampler [[Google Drive](https://drive.google.com/file/d/1kezDpwqA4a3WGq5PfS1kDrASf3-bN4js/view?usp=sharing)] pretrained on MS-Celeb-1M.
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
Here we use 'NIR' and 'VIS' in the training list to distinguish the modalities of images. If your list has other distinguishable marks,
please change them correspondingly in `./data/dataset.py` (lines 28, 38, 66, and 68).
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

@article{fu2022towards,
  title={Towards Lightweight Pixel-Wise Hallucination for Heterogeneous Face Recognition},
  author={Fu, Chaoyou and Zhou, Xiaoqiang and He, Weizan and He, Ran},
  journal={IEEE TPAMI},
  year={2022}
}

@inproceedings{duan2020cross,
  title={Cross-spectral face hallucination via disentangling independent factors},
  author={Duan, Boyan and Fu, Chaoyou and Li, Yi and Song, Xingguang and He, Ran},
  booktitle={CVPR},
  year={2020}
}

```








