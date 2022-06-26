import torch
from dataset import A_B_Dataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from discriminator_model import Discriminator
from generator_model import Generator
from sklearn.manifold import TSNE
from vgg_new import VGG
from torchvision import transforms
from vgg_new import get_vgg_layers
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
import seaborn as sns
import pandas as pd
from math import sqrt
from statistics import mean
import argparse
from torchvision import models
import os

args = {}

def dataset_name():
    return args.dataset + "/"

def data_dir_path():
    return "dataset/" + dataset_name()

def train_dir_a():
    return data_dir_path() + "trainA/"

def train_dir_b():
    return data_dir_path() + "trainB/"

def val_dir():
    return data_dir_path() + "test"

def trained_model_path():
    return args.trained_model

def trained_GAN_path():
    return args.pretrained_GAN + "/"

def create_output_folder():
    try:
        output_folder = output_image_path()
        os.makedirs(output_folder)
        print("Successfully created output folder ", output_folder)
    except OSError:
        print(OSError)
    return output_folder

def output_image_path():
    return "cyclegan_output_" + args.dataset + "/"

def train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
    A_reals = 0
    A_fakes = 0
    loop = tqdm(loader, leave=True)
    
    #train loop
    for idx, (B, A) in enumerate(loop):
        
        #print("idx: ", idx)
        B = B.to(config.DEVICE)
        A = A.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_A = gen_A(B)
            D_A_real = disc_A(A)
            D_A_fake = disc_A(fake_A.detach())
            A_reals += D_A_real.mean().item()
            A_fakes += D_A_fake.mean().item()
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            fake_B = gen_B(A)
            D_B_real = disc_B(B)
            D_B_fake = disc_B(fake_B.detach())
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            # put it togethor
            D_loss = (D_A_loss + D_B_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # ADVERARIAL LOSS for both generators
            D_A_fake = disc_A(fake_A)
            D_B_fake = disc_B(fake_B)
            
            #print("Disc Shapes : ", D_A_fake.shape, D_B_fake.shape)

            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

            # CYCLE LOSS
            
            cycle_B = gen_B(fake_A)
            cycle_A = gen_A(fake_B)
            cycle_B_loss = l1(B, cycle_B)
            cycle_A_loss = l1(A, cycle_A)
            #print("Original Shapes : ", A.shape, B.shape, cycle_A.shape, cycle_B.shape)
            
                    
            # IDENTITY LOSS (remove these for efficiency if you set lambda_identity=0)
            identity_B = gen_B(B)
            identity_A = gen_A(A)
            identity_B_loss = l1(B, identity_B)
            identity_A_loss = l1(A, identity_A)

            # add all togethor
            G_loss = (
                loss_G_B
                + loss_G_A
                + cycle_B_loss * config.LAMBDA_CYCLE
                + cycle_A_loss * config.LAMBDA_CYCLE
                + identity_A_loss * config.LAMBDA_IDENTITY
                + identity_B_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 100 == 0:            
            img_A = output_image_path() + 'A_' + str(epoch) + '_'+ str(idx) + '.png'
            img_B = output_image_path() + 'B_' + str(epoch) + '_'+ str(idx) + '.png'
            #save_image(fake_A*0.5+0.5, img_A)
            #save_image(fake_B*0.5+0.5, img_B)
            
            # generate samples A->B, B->A
            fakeB = gen_B(A)*0.5+0.5
            fakeA = gen_A(B)*0.5+0.5
            
            new_im1 = torch.cat([A, fakeB], dim = 0)
            new_im2 = torch.cat([B, fakeA], dim = 0)
            
            grid_img1 = make_grid(new_im1, nrow=2)
            grid_img2 = make_grid(new_im2, nrow=2)
            save_image( grid_img1, img_A)
            save_image( grid_img2 , img_B)
            
            #save_image(fake_A*0.5+0.5, f"saved_images/A_{idx}.png")
            #save_image(fake_B*0.5+0.5, f"saved_images/B_{idx}.png")

        loop.set_postfix(A_real=A_reals/(idx+1), A_fake=A_fakes/(idx+1))
  
    
    cl_Ao, cl_Bo,  = cycle_A_loss.item(), cycle_B_loss.item()
    d_Ao, d_Bo =  D_A_loss.item(), D_B_loss.item()
    g_Ao, g_Bo =  loss_G_A.item(), loss_G_B.item()
    

    
    return cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo,

def main():
    #torch.cuda.set_device(args.gpu)
    output_folder = create_output_folder()
    #print("output image path :", output_image_path())
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_B.parameters()) + list(gen_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if args.load_model:
        load_checkpoint(
            trained_GAN_path()+config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            trained_GAN_path()+config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            trained_GAN_path()+config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            trained_GAN_path()+config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, config.LEARNING_RATE,
        )


    dataset = A_B_Dataset(
        root_A=train_dir_a(), root_B=train_dir_b(), transform=config.transforms
    )
    #val_dataset = A_B_Dataset(
    #   root_A=config.VAL_DIR+"A", root_B=config.VAL_DIR+"B", transform=config.transforms
    #)

    #val_loader = DataLoader(
    #    val_dataset,
    #    batch_size=1,
    #    shuffle=False,
    #    pin_memory=True,
    #)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    cl_A = []
    cl_B = []
    d_A = []
    d_B = []
    g_A = []
    g_B = []


    for epoch in range(config.NUM_EPOCHS):
        print("\n For epoch :", epoch)
        cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo = train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch)
        print("Loss : ",cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo)
        cl_A.append(cl_Ao)
        cl_B.append(cl_Bo)
        d_A.append(d_Ao)
        d_B.append(d_Bo)
        g_A.append(g_Ao)
        g_B.append(g_Bo)
        

        if config.SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=output_image_path() + config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=output_image_path() + config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=output_image_path() + config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=output_image_path() + config.CHECKPOINT_CRITIC_B)

    # plots
    plt.figure(figsize=(10, 7))
    plt.plot(cl_A, color='green', linestyle='-', label='Cycle_loss_A')
    plt.plot(cl_B, color='yellow', linestyle='-', label='Cycle_loss_B')
    plt.plot(d_A, color='blue', linestyle='-', label='Disc A')
    plt.plot(d_B, color='red', linestyle='-', label='Disc B')
    plt.plot(g_A, color='orange', linestyle='-', label='Gen A')
    plt.plot(g_B, color='black', linestyle='-', label='Gen B')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig_name = output_folder + "_tsne.jpg"
    plt.savefig(fig_name)
    #plt.show()
    
 
    
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Unpaired Image-to-Image Translation using TSNE with cycle gan")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu index")
    parser.add_argument("--dataset", type=str, default="horse2zebra", help="dataset name")
    parser.add_argument("--load_model", type=bool, default=False, help="load GAN model pretrained")
    parser.add_argument("--pretrained_GAN", type=str, default="", help="pretrained GAN")

    parser.add_argument("--numEpochs", type=int, default=100, help="number of epochs")
    global args
    args = parser.parse_args()
    print(args)

if __name__ == "__main__":
    parse_arguments()
    main()

# -*- coding: utf-8 -*-

