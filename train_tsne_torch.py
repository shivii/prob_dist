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
import train_options

def train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
    A_reals = 0
    A_fakes = 0
    loop = tqdm(loader, leave=True)
    
    #print("\n loop:", loop)
    real = torch.tensor([[1],])
    fake = torch.tensor([[0],])
    #print("shape real", real.shape)
    
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

        #if idx % 80 == 0:
            #img_A = 'tsne_images/genA_' + str(idx) + '_'+ str(epoch) + '.png'
            #img_B = 'tsne_images/genB_' + str(idx) + '_'+ str(epoch) + '.png'
            #save_image(fake_A*0.5+0.5, img_A)
            #save_image(fake_B*0.5+0.5, img_B)
            #save_image(fake_A*0.5+0.5, f"saved_images/A_{idx}.png")
            #save_image(fake_B*0.5+0.5, f"saved_images/B_{idx}.png")

        loop.set_postfix(A_real=A_reals/(idx+1), A_fake=A_fakes/(idx+1))
    
   
    cl_Ao, cl_Bo,  = cycle_A_loss.item(), cycle_B_loss.item()
    d_Ao, d_Bo =  D_A_loss.item(), D_B_loss.item()
    g_Ao, g_Bo =  loss_G_A.item(), loss_G_B.item()
    con_Ao, con_Bo =  0, 0
    g_losses =  G_loss.item()
    d_losses = D_loss.item()
        
    return cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo, con_Ao, con_Bo, g_losses, d_losses

if __name__ == "__main__":
    opt = train_options.TrainOptions().parse() 
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

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, config.LEARNING_RATE,
        )


    dataset = A_B_Dataset(
        root_A = opt.dataroot/ + 'trainA/', root_B = opt.dataroot/ + 'trainB/', transform=config.transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads),
        pin_memory=True
    )

    print("loaded !!!")
    """
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    cl_A = []
    cl_B = []
    d_A = []
    d_B = []
    g_A = []
    g_B = []
    con_A = []
    con_B = []
    g_loss = []
    d_loss = []

    for epoch in range(config.NUM_EPOCHS):
        print("\n For epoch :", epoch)
        cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo, con_Ao, con_Bo, gen_loss, dis_loss = train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch)
        print("Loss : ",cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo, con_Ao, con_Bo, gen_loss)
        cl_A.append(cl_Ao)
        cl_B.append(cl_Bo)
        d_A.append(d_Ao)
        d_B.append(d_Bo)
        g_A.append(g_Ao)
        g_B.append(g_Bo)
        con_A.append(con_Ao)
        con_B.append(con_Bo)
        g_loss.append(gen_loss)
        d_loss.append(dis_loss)

        if config.SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)

    ## plots
    ##plt.figure(figsize=(10, 7))
    #plt.plot(cl_A, color='green', linestyle='-', label='Cycle_loss_A')
    #plt.plot(cl_B, color='yellow', linestyle='-', label='Cycle_loss_B')
    #plt.plot(d_A, color='blue', linestyle='-', label='Disc A')
    #plt.plot(d_B, color='red', linestyle='-', label='Disc B')
    #plt.plot(g_A, color='orange', linestyle='-', label='Gen A')
    #plt.plot(g_B, color='black', linestyle='-', label='Gen B')
    #plt.plot(list_distA, color='pink', linestyle='-', label='TSNE')
    #plt.plot(list_distB, color='pink', linestyle='-', label='TSNE')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.savefig('tsne.jpg')
    #plt.show()
    #
    """



