import numpy as np
import os, random
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class Real_Dataset(data.Dataset):
    def __init__(self, args):
        super(Real_Dataset, self).__init__()

        self.img_root = args.img_root_A
        self.img_list = self.list_reader(args.train_list_A)

        self.transform = transforms.Compose([
            transforms.RandomCrop(128),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_name, label = self.img_list[index]
        img_path = os.path.join(self.img_root, img_name)
        img = Image.open(img_path).convert('L')
        img = self.transform(img)

        return {'img': img, 'label': int(label)}

    def __len__(self):
        return len(self.img_list)

    def list_reader(self, list_file):
        img_list = []
        with open(list_file, 'r') as f:
            for line in f.readlines():
                img_name, label = line.strip().split(' ')
                img_list.append((img_name, label))
        return img_list


class Mix_Dataset(data.Dataset):
    def __init__(self, args):
        super(Mix_Dataset, self).__init__()

        self.img_root_A = args.img_root_A
        self.img_root_B = args.img_root_B

        self.list_file_A = args.train_list_A
        self.list_file_B = args.train_list_B

        self.transform = transforms.Compose([
            transforms.RandomCrop(128),
            transforms.ToTensor()
        ])

        self.img_list_A, self.img_list_B = self.list_reader()

    def __getitem__(self, index):
        img_name, label = self.img_list_A[index]
        img_path_A = os.path.join(self.img_root_A, img_name)
        img_A = Image.open(img_path_A).convert('L')
        img_A = self.transform(img_A)

        img_name = self.img_list_B[index]
        img_path_B = os.path.join(self.img_root_B, img_name)
        img_B = Image.open(img_path_B).convert('L')
        img_B = self.transform(img_B)

        img_path_B_pair = img_path_B.replace('gen_images/nir', 'gen_images/vis')
        img_B_pair = Image.open(img_path_B_pair).convert('L')
        img_B_pair = self.transform(img_B_pair)

        return {'img_A': img_A, 'label': int(label),
                'img_B': img_B, 'img_B_pair': img_B_pair}

    def __len__(self):
        return len(self.img_list_B)

    def list_reader(self):
        img_list_A = []
        with open(self.list_file_A) as f:
            img_names = f.readlines()
            for img_name in img_names:
                img_name, label = img_name.strip().split(' ')
                img_list_A.append((img_name, label))

        img_list_B = []
        with open(self.list_file_B) as f:
            img_names = f.readlines()
            for img_name in img_names:
                img_name = img_name.strip()
                img_list_B.append(img_name)

        rep_num = int(len(img_list_B) / len(img_list_A))
        img_list_A_extend = []
        for i in range(rep_num):
            img_list_A_extend.extend(img_list_A)

        res_num = len(img_list_B) - rep_num * len(img_list_A)
        for i in range(res_num):
            img_list_A_extend.append(img_list_A[i])

        return img_list_A_extend, img_list_B
