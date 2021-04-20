import numpy as np
import os, random
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class Dataset(data.Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.img_root = args.img_root
        self.list_file = args.train_list

        self.transform = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.ToTensor()
        ])

        self.img_list, self.pair_dict = self.list_reader()

    def __getitem__(self, index):
        img_name, label = self.img_list[index]

        if 'NIR' in img_name:
            img_path = os.path.join(self.img_root, img_name)
            nir = Image.open(img_path)
            nir = self.transform(nir)

            # paired vis
            img_name = self.get_pair(label, 'VIS')
            img_path = os.path.join(self.img_root, img_name)
            vis = Image.open(img_path)
            vis = self.transform(vis)
        elif 'VIS' in img_name:
            img_path = os.path.join(self.img_root, img_name)
            vis = Image.open(img_path)
            vis = self.transform(vis)

            # paired nir
            img_name = self.get_pair(label, 'NIR')
            img_path = os.path.join(self.img_root, img_name)
            nir = Image.open(img_path)
            nir = self.transform(nir)

        return {'nir': nir, 'vis': vis, 'label': int(label)}

    def __len__(self):
        return len(self.img_list)

    def list_reader(self):
        def dict_profile():
            return {'NIR': [], 'VIS': []}

        img_list = []
        pair_dict = defaultdict(dict_profile)
        with open(self.list_file) as f:
            img_names = f.readlines()
            for img_name in img_names:
                img_name, label = img_name.strip().split(' ')
                img_list.append((img_name, label))

                if 'NIR' in img_name:
                    pair_dict[label]['NIR'].append(img_name)
                elif 'VIS' in img_name:
                    pair_dict[label]['VIS'].append(img_name)

        # find IDs with paired data
        img_list_update = []
        for img_name, label in img_list:
            if (len(pair_dict[label]['NIR']) > 0) and (len(pair_dict[label]['VIS']) > 0):
                img_list_update.append((img_name, label))

        return img_list_update, pair_dict

    def get_pair(self, label, modality):
        img_name = random.choice(self.pair_dict[label][modality])
        return img_name
