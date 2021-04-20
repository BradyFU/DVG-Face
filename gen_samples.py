import os
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable

from utils import *
from network.ID_net import define_ID
from network.G_net import define_G
from data.dataset import Dataset


parser = argparse.ArgumentParser()

parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--output_path_nir', default='gen_images/nir', type=str)
parser.add_argument('--output_path_vis', default='gen_images/vis', type=str)

parser.add_argument('--weights_dec', default='./pre_train/dec_epoch_45.pth.tar', type=str, help='dec is the identity sampler')
parser.add_argument('--weights_encoder_nir', default='./model/encoder_nir_epoch_5.pth.tar', type=str)
parser.add_argument('--weights_encoder_vis', default='./model/encoder_vis_epoch_5.pth.tar', type=str)
parser.add_argument('--weights_decoder', default='./model/decoder_epoch_5.pth.tar', type=str)

parser.add_argument('--img_root',  default='', type=str)
parser.add_argument('--train_list', default='', type=str)


def main():
    global opt, model
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    cudnn.benchmark = True

    if not os.path.exists(args.output_path_nir):
        os.makedirs(args.output_path_nir)

    if not os.path.exists(args.output_path_vis):
        os.makedirs(args.output_path_vis)

    # id sampler
    dec = define_ID()
    load_model(dec, args.weights_dec)
    set_requires_grad([dec], False)
    dec.eval()

    # generator
    encoder_nir, encoder_vis, decoder = define_G(input_dim=3, output_dim=3, ndf=32)
    load_model(encoder_nir, args.weights_encoder_nir)
    load_model(encoder_vis, args.weights_encoder_vis)
    load_model(decoder, args.weights_decoder)

    set_requires_grad([encoder_nir, encoder_vis, decoder], False)
    encoder_nir.eval()
    encoder_vis.eval()
    decoder.eval()

    # dataset
    train_loader = torch.utils.data.DataLoader(
        Dataset(args), batch_size=50, shuffle=True, num_workers=8, pin_memory=True)

    img_num = 0
    list_file = open(args.output_path_nir.split("/")[0] + "/img_list.txt", "w")
    for epoch in range(1, 100000):
        for iteration, data in enumerate(train_loader):
            nir = Variable(data["nir"].cuda())
            vis = Variable(data["vis"].cuda())

            batch_size = nir.size(0)
            noise = torch.zeros(batch_size, 256).normal_(0, 1).cuda()
            id_noise = dec(noise)

            z_nir = encoder_nir(nir, "enc")
            z_vis = encoder_vis(vis, "enc")

            style_nir = encoder_nir(z_nir, "style")
            style_vis = encoder_vis(z_vis, "style")

            assign_adain_params(style_nir, decoder)
            fake_nir = decoder(torch.cat([id_noise, z_nir], dim=1), "nir")

            assign_adain_params(style_vis, decoder)
            fake_vis = decoder(torch.cat([id_noise, z_vis], dim=1), "vis")

            # save images
            fake_nir = fake_nir.data.cpu().numpy()
            fake_vis = fake_vis.data.cpu().numpy()
            for i in range(batch_size):
                img_num = img_num + 1
                list_file.write(str(img_num) + ".jpg" + "\n")
                print(img_num)

                save_img = fake_nir[i, :, :, :]
                save_img = np.transpose((255 * save_img).astype("uint8"), (1, 2, 0))
                output = Image.fromarray(save_img)
                save_name = str(img_num) + ".jpg"
                output.save(os.path.join(args.output_path_nir, save_name))

                save_img = fake_vis[i, :, :, :]
                save_img = np.transpose((255 * save_img).astype("uint8"), (1, 2, 0))
                output = Image.fromarray(save_img)
                save_name = str(img_num) + ".jpg"
                output.save(os.path.join(args.output_path_vis, save_name))

                if img_num == 100000:
                    print("we have generated 100k paired images")
                    list_file.close()
                    exit(0)




if __name__ == "__main__":
    main()