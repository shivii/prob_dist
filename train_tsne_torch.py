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
from tsne_torch import TorchTSNE as TSNE
from torchvision import models

def dataset_name():
    return "horse2zebra/"

def data_dir_path():
    return "dataset/" + dataset_name()

def train_dir_a():
    return data_dir_path() + "trainA/"

def train_dir_b():
    return data_dir_path() + "trainB/"

def val_dir():
    return data_dir_path() + "test"

def trained_model_path():
    return "vgg19_23_06.pt"


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
        #                        512, 512, 'M', 512, 512, 512, 512, 'M']
        #vgg19_layers = get_vgg_layers(vgg19_config, batch_norm=True)
        model = models.vgg19()
        
        #print(model)
        model.load_state_dict(torch.load(trained_model_path()))
        self.vgg_ft = model.features.eval().to(config.DEVICE)
        self.avgpool = nn.AdaptiveAvgPool2d(7).to(config.DEVICE)
        self.vgg_fcLayers = nn.Sequential(
                                *list(model.classifier.children())[:-3]).to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg_ft.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False
        for param in self.vgg_fcLayers.parameters():
            param.requires_grad = False

    def forward(self, input):
        x = self.vgg_ft(input)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.vgg_fcLayers(h)

        return x

def get_tsne(data, labels, i):  
    n_components = 2
    tsne = manifold.TSNE(n_components = n_components, random_state = 0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data

def get_tsne_torch(data, labels, i):
    data_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(data)
    return data_emb
    
def plot_representations(tx, ty, labels, i, name):
    classes = [1,0]
    # initialize a matplotlib plot
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # for every class, we'll add a scatter plot separately
    for label in classes:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]
    
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
    
        # convert the class color to matplotlib format
        #color = np.array(classes[label], dtype=np.float) / 255
        #color = range(len(targets))
    
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, label=label)
    
    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plot_name = "tsne_projections/projection_" + str(i) + name
    plt.savefig(plot_name)    
    # finally, show the plot
    #plt.show()

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def tsne_loss(tx, ty):
    distances = []
    for i in range(0, len(tx), 2):
        p1 = (tx[i], ty[i]) # first point
        p2 = (tx[i+1], ty[i+1]) # second point
        dist = sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2) # Pythagorean theorem
        distances.append(dist)
    distance = mean(distances)
    return distance


def train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch, vgg_ft, tsne_distance_a, tsne_distance_b):
    A_reals = 0
    A_fakes = 0
    loop = tqdm(loader, leave=True)
    
    #print("\n loop:", loop)
    real = torch.tensor([[1],])
    fake = torch.tensor([[0],])
    #print("shape real", real.shape)
    
    # initialize various vectors
    tsne_ftA = torch.zeros((0, 4096), dtype=torch.float32)
    
    tsne_embeddingsA = torch.zeros((0, 4096), dtype=torch.float32)
    
    imagesA = torch.zeros((0, 65536), dtype=torch.float32)
    
    labels_A = torch.zeros((0,1), dtype=torch.uint8)
    
    tsne_ftB = torch.zeros((0, 4096), dtype=torch.float32)
    
    tsne_embeddingsB = torch.zeros((0, 4096), dtype=torch.float32)
    
    imagesB = torch.zeros((0, 65536), dtype=torch.float32)
    
    labels_B = torch.zeros((0,1), dtype=torch.uint8)
    
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
            
            # TSNE metric    
            # dimentionality reduction from 4D(1,3,256,256) to (256,256) 
            
            # get features and embedding from VGG19---------------------------------------- ###
            
            featA = vgg_ft(A)
            featB = vgg_ft(B)
            featCycleA = vgg_ft(cycle_A)
            featCycleB = vgg_ft(cycle_B)
            
            # tsne for B, Cycle B
            # tsne for A, Cycle A    
            
            tsne_embeddingsA = torch.cat((tsne_embeddingsA, featA.detach().cpu()), 0)
            labels_A = torch.cat((labels_A, real),0)
            tsne_embeddingsA = torch.cat((tsne_embeddingsA, featCycleA.detach().cpu()), 0)
            labels_A = torch.cat((labels_A, fake),0)
            
            tsne_embeddingsB = torch.cat((tsne_embeddingsB, featB.detach().cpu()), 0)
            labels_B = torch.cat((labels_B, real),0)
            tsne_embeddingsB = torch.cat((tsne_embeddingsB, featCycleB.detach().cpu()), 0)
            labels_B = torch.cat((labels_B, fake),0)
            
            #print("A vgg features shape:", tsne_embeddingsA.shape)
            #print("B vgg features shape:", tsne_embeddingsB.shape)
            #print("A vgg labels shape:", labels_A.shape)
            #print("B vgg labels shape:", labels_B.shape)
            
            #### --------------------------------------------------------------------------- ###
            """
            A_im = transforms.Grayscale(num_output_channels=1)(A)
            cycleA_im = transforms.Grayscale(num_output_channels=1)(cycle_A)
            A_im = torch.flatten(A_im)
            A_im = torch.unsqueeze(A_im, 0)
            cycleA_im = torch.flatten(cycleA_im)
            cycleA_im = torch.unsqueeze(cycleA_im, 0)
            #print("Grayscale 2Shapes A: ", A_im.shape, cycleA_im.shape,)
            #print("Sqeeze 2Shapes A: ", A_im.shape, cycleA_im.shape,)

            B_im = transforms.Grayscale(num_output_channels=1)(B)            
            cycleB_im = transforms.Grayscale(num_output_channels=1)(cycle_B)
            B_im = torch.flatten(B_im)
            B_im = torch.unsqueeze(B_im, 0)
            cycleB_im = torch.flatten(cycleB_im)
            cycleB_im = torch.unsqueeze(cycleB_im, 0)
            #print("Grayscale 2Shapes B: ", B_im.shape, cycleB_im.shape,)
            #print("Sqeeze 2Shapes B: ", B_im.shape, cycleB_im.shape,)

            #cycleB_im = transforms.Grayscale(num_output_channels=1)(cycleB_im)


            imagesA = torch.cat((imagesA, A_im.detach().cpu()), 0)
            labels_A = torch.cat((labels_A, real),0)
            imagesA = torch.cat((imagesA, cycleA_im.detach().cpu()), 0)
            labels_A = torch.cat((labels_A, fake),0)
            
            
            imagesB = torch.cat((imagesB, B_im.detach().cpu()), 0)
            labels_B = torch.cat((labels_B, real),0)
            imagesB = torch.cat((imagesB, cycleB_im.detach().cpu()), 0)
            labels_B = torch.cat((labels_B, fake),0)
            """
            #print("image A:", imagesA.shape)
            #print("image B:", imagesB.shape)
            
            # IDENTITY LOSS (remove these for efficiency if you set lambda_identity=0)
            #identity_B = gen_B(B)
            #identity_A = gen_A(A)
            #identity_B_loss = l1(B, identity_B)
            #identity_A_loss = l1(A, identity_A)

            # add all togethor
            G_loss = (
                loss_G_B
                + loss_G_A
                + cycle_B_loss * config.LAMBDA_CYCLE
                + cycle_A_loss * config.LAMBDA_CYCLE
                + tsne_distance_a * 10
                + tsne_distance_b * 10
                #+ identity_A_loss * config.LAMBDA_IDENTITY
                #+ identity_B_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 80 == 0:
            img_A = 'tsne_images/genA_' + str(idx) + '_'+ str(epoch) + '.png'
            img_B = 'tsne_images/genB_' + str(idx) + '_'+ str(epoch) + '.png'
            save_image(fake_A*0.5+0.5, img_A)
            save_image(fake_B*0.5+0.5, img_B)
            #save_image(fake_A*0.5+0.5, f"saved_images/A_{idx}.png")
            #save_image(fake_B*0.5+0.5, f"saved_images/B_{idx}.png")

        loop.set_postfix(A_real=A_reals/(idx+1), A_fake=A_fakes/(idx+1))
    
    """
    # COMPUTE TSNE FOR A
    tsne_embeddingsA = np.array(imagesA)
        
    tsne_dataA = get_tsne(tsne_embeddingsA, labels_A, epoch)
    tsne_dataA = scale_to_01_range(tsne_dataA)
    
    txA = tsne_dataA[:,0]
    tyA = tsne_dataA[:,1]
    
    plot_representations(txA, tyA, labels_A, epoch)

    distanceA = tsne_loss(txA, tyA)
    
    # COMPUTE TSNE FOR B
    tsne_embeddingsB = np.array(imagesB)
        
    tsne_dataB = get_tsne(tsne_embeddingsB, labels_B, epoch)
    tsne_dataB = scale_to_01_range(tsne_dataB)
    
    txB = tsne_dataB[:,0]
    tyB = tsne_dataB[:,1]
    
    plot_representations(txB, tyB, labels_B, epoch)

    distanceB = tsne_loss(txB, tyB)
    
    """
    # COMPUTE TSNE FOR A
    tsne_ftA= np.array(tsne_embeddingsA)
    print("tsne ft A:", tsne_ftA.shape)
        
    tsne_dataA = get_tsne_torch(tsne_ftA, labels_A, epoch)
    tsne_dataA = scale_to_01_range(tsne_dataA)
    
    txA = tsne_dataA[:,0]
    tyA = tsne_dataA[:,1]
    
    plot_representations(txA, tyA, labels_A, epoch, "A")

    distanceA = tsne_loss(txA, tyA)
    
    # COMPUTE TSNE FOR B
    tsne_ftB= np.array(tsne_embeddingsB)
    print("tsne ft B:", tsne_ftB.shape)
        
    tsne_dataB = get_tsne_torch(tsne_embeddingsB, labels_B, epoch)
    tsne_dataB = scale_to_01_range(tsne_dataB)
    
    txB = tsne_dataB[:,0]
    tyB = tsne_dataB[:,1]
    
    plot_representations(txB, tyB, labels_B, epoch, "B")

    distanceB = tsne_loss(txB, tyB)

    
    cl_Ao, cl_Bo,  = cycle_A_loss.item(), cycle_B_loss.item()
    d_Ao, d_Bo =  D_A_loss.item(), D_B_loss.item()
    g_Ao, g_Bo =  loss_G_A.item(), loss_G_B.item()
    con_Ao, con_Bo =  0, 0
    g_losses =  G_loss.item()
    d_losses = D_loss.item()
    

    
    return cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo, con_Ao, con_Bo, g_losses, d_losses, distanceA, distanceB

def main():
    torch.cuda.set_device(1)
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
    con_A = []
    con_B = []
    g_loss = []
    d_loss = []
    vgg_loss = VGGLoss()
    distanceA = 0
    distanceB = 0

    list_distA = []
    list_distB = []


    for epoch in range(config.NUM_EPOCHS):
        print("\n For epoch :", epoch)
        cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo, con_Ao, con_Bo, gen_loss, dis_loss, distanceA, distanceB = train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch, vgg_loss, distanceA, distanceB)
        print("Loss : ",cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo, con_Ao, con_Bo, gen_loss, distanceA, distanceB)
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
        list_distA.append(distanceA)
        list_distB.append(distanceB)

        if config.SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)

    # plots
    plt.figure(figsize=(10, 7))
    plt.plot(cl_A, color='green', linestyle='-', label='Cycle_loss_A')
    plt.plot(cl_B, color='yellow', linestyle='-', label='Cycle_loss_B')
    plt.plot(d_A, color='blue', linestyle='-', label='Disc A')
    plt.plot(d_B, color='red', linestyle='-', label='Disc B')
    plt.plot(g_A, color='orange', linestyle='-', label='Gen A')
    plt.plot(g_B, color='black', linestyle='-', label='Gen B')
    plt.plot(list_distA, color='pink', linestyle='-', label='TSNE')
    plt.plot(list_distB, color='pink', linestyle='-', label='TSNE')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('tsne.jpg')
    plt.show()
    
    
if __name__ == "__main__":
    main()
