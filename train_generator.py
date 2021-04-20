import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from utils import *
from network.ID_net import define_ID
from network.G_net import define_G
from network.lightcnn import LightCNN_29v2
from data.dataset import Dataset


parser = argparse.ArgumentParser()

parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--pre_epoch', default=0, type=int, help='train from previous model')

parser.add_argument('--print_iter', default=20, type=int, help='print frequency')
parser.add_argument('--save_epoch', default=1, type=int)
parser.add_argument('--output_path', default='./results', type=str)

parser.add_argument('--weights_lightcnn', default='./pre_train/LightCNN_29Layers_V2_checkpoint.pth.tar', type=str)
parser.add_argument('--weights_dec', default='./pre_train/dec_epoch_45.pth.tar', type=str, help='dec is the identity sampler')
parser.add_argument('--img_root',  default='', type=str)
parser.add_argument('--train_list', default='', type=str)


def main():
    global args
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    cudnn.benchmark = True

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # lightcnn
    LightCNN = LightCNN_29v2(is_train=False)
    print("=> loading pretrained lightcnn '{}'".format(args.weights_lightcnn))
    load_model(LightCNN, args.weights_lightcnn)
    set_requires_grad([LightCNN], False)
    LightCNN.eval()

    # id sampler
    dec = define_ID()
    print("=> loading pretrained identity sampler '{}'".format(args.weights_dec))
    load_model(dec, args.weights_dec)
    set_requires_grad([dec], False)
    dec.eval()

    # generator
    encoder_nir, encoder_vis, decoder = define_G(input_dim=3, output_dim=3, ndf=32)

    # load pretrained model
    if args.pre_epoch:
        print("load pretrained model %d" % args.pre_epoch)
        load_model(encoder_nir, "./model/encoder_nir_epoch_%d.pth.tar" % args.pre_epoch)
        load_model(encoder_vis, "./model/encoder_vis_epoch_%d.pth.tar" % args.pre_epoch)
        load_model(decoder, "./model/decoder_epoch_%d.pth.tar" % args.pre_epoch)

    # dataset
    train_loader = torch.utils.data.DataLoader(
        Dataset(args), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # optimizer
    optimizer = optim.Adam(list(encoder_nir.parameters()) + list(encoder_vis.parameters()) +
                           list(decoder.parameters()), lr=args.lr, betas=(0.5, 0.999))

    # criterion
    criterionPix = torch.nn.L1Loss().cuda()

    # train
    start_epoch = args.pre_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):

        # creat random index
        arange = torch.arange(args.batch_size).cuda()
        idx = torch.randperm(args.batch_size).cuda()
        while 0.0 in (idx - arange):
            idx = torch.randperm(args.batch_size).cuda()

        for iteration, data in enumerate(train_loader, start=1):
            # get data
            nir = Variable(data["nir"].cuda())
            vis = Variable(data["vis"].cuda())

            batch_size = nir.size(0)
            if batch_size < args.batch_size:
                continue

            id_vis = LightCNN(rgb2gray(vis))
            noise = torch.zeros(batch_size, 256).normal_(0, 1).cuda()
            id_noise = dec(noise)

            # forward
            z_nir = encoder_nir(nir, "enc")
            z_vis = encoder_vis(vis, "enc")

            style_nir = encoder_nir(z_nir, "style")
            style_vis = encoder_vis(z_vis, "style")

            assign_adain_params(style_nir, decoder)
            rec_nir = decoder(torch.cat([id_vis, z_nir], dim=1), "nir")
            rec_nir_idx = decoder(torch.cat([id_vis[idx, :], z_nir], dim=1), "nir")
            fake_nir = decoder(torch.cat([id_noise, z_nir], dim=1), "nir")

            assign_adain_params(style_vis, decoder)
            rec_vis = decoder(torch.cat([id_vis, z_vis], dim=1), "vis")
            rec_vis_idx = decoder(torch.cat([id_vis[idx, :], z_vis], dim=1), "vis")
            fake_vis = decoder(torch.cat([id_noise, z_vis], dim=1), "vis")

            # orthogonal loss
            loss_ort = 50 * (ort_loss(z_nir, id_vis) + ort_loss(z_vis, id_vis))

            # pixel loss
            loss_pix = 100 * ((criterionPix(rec_nir, nir) + criterionPix(rec_vis, vis)) +
                               0.1 * (criterionPix(rec_nir_idx, nir) + criterionPix(rec_vis_idx, vis)) +
                               0.1 * (criterionPix(fake_nir, nir) + criterionPix(fake_vis, vis)))

            # identity preserving loss
            id_nir_rec = LightCNN(rgb2gray(rec_nir))
            id_vis_rec = LightCNN(rgb2gray(rec_vis))
            id_nir_fake = LightCNN(rgb2gray(fake_nir))
            id_vis_fake = LightCNN(rgb2gray(fake_vis))

            real_ang_rec = ang_loss(id_nir_rec, id_vis) + ang_loss(id_vis_rec, id_vis)
            real_ang_pair = ang_loss(id_nir_rec, id_vis_rec)

            fake_ang_rec = ang_loss(id_nir_fake, id_noise) + ang_loss(id_vis_fake, id_noise)
            fake_ang_pair = ang_loss(id_nir_fake, id_vis_fake)

            loss_ip = - 0.1 * (real_ang_rec + 0.05 * real_ang_pair + fake_ang_rec + 0.05 * fake_ang_pair)

            # all losses
            loss = loss_ort + loss_pix + loss_ip

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print log
            if iteration % args.print_iter == 0:
                info = "====> Epoch[{}][{}/{}] | ".format(epoch, iteration, len(train_loader))
                info += "Loss: pix: {:4.2f} ort: {:4.2f} | Ang-real rec: {:4.2f} pair: {:4.2f} | Ang-fake rec: {:4.2f} pair: {:4.2f}".format(
                    loss_pix.item(), loss_ort.item(), real_ang_rec.item(), real_ang_pair.item(), fake_ang_rec.item(), fake_ang_pair.item())
                print(info)

            # save images
            if iteration % 500 == 0:
                vutils.save_image(torch.cat([nir, rec_nir, rec_nir_idx, fake_nir, nir[idx, :],
                                             vis, rec_vis, rec_vis_idx, fake_vis, vis[idx, :]], dim=0).data,
                                  "{}/Epoch_{:03d}_Iter_{:06d}_img.png".format(args.output_path, epoch, iteration), nrow=batch_size)

        # save model
        if epoch % args.save_epoch == 0:
            save_checkpoint(encoder_nir, epoch, "encoder_nir")
            save_checkpoint(encoder_vis, epoch, "encoder_vis")
            save_checkpoint(decoder, epoch, "decoder")




if __name__ == "__main__":
    main()
