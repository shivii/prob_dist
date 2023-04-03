import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import copy
import random
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
                512, 'M']

vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M']

vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
                512, 512, 'M', 512, 512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

def get_vgg_layers(config, batch_norm):

    layers = []
    in_channels = 3

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):

        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device

        torch.save(model.state_dict(), 'init_params.pt')

    def range_test(self, iterator, end_lr=10, num_iter=100,
                   smooth_f=0.05, diverge_th=5):

        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)

        iterator = IteratorWrapper(iterator)

        for iteration in range(num_iter):

            loss = self._train_batch(iterator)

            lrs.append(lr_scheduler.get_last_lr()[0])

            # update lr
            lr_scheduler.step()

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            if loss < best_loss:
                best_loss = loss

            losses.append(loss)

            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break

        # reset model to initial parameters
        model.load_state_dict(torch.load('init_params.pt'))

        return lrs, losses

    def _train_batch(self, iterator):

        self.model.train()

        self.optimizer.zero_grad()

        x, y = iterator.get_batch()

        x = x.to(self.device)
        y = y.to(self.device)

        y_pred, _ = self.model(x)

        loss = self.criterion(y_pred, y)

        loss.backward()

        self.optimizer.step()

        return loss.item()


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in
                self.base_lrs]


class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()
#    print("-->>>starting training")
    for (x, y) in tqdm(iterator, desc="Training", leave=False):
#        print("-->> inside training for loop")
        x = x.to(device)
#        print("-->> loaded x")
        y = y.to(device)
#        print("-->> loaded y")
        optimizer.zero_grad()
#        print("--> optimizer run")
        y_pred, _ = model(x)
#        print("-->> pred run")
        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)
        #print("acc:")
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



class SquarePad:
        def __call__(self, image):
            max_wh = max(image.size)
            p_left, p_top = [(max_wh - s) // 2 for s in image.size]
            p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
            padding = (p_left, p_top, p_right, p_bottom)
            return F.pad(image, padding, 255, 'constant') 

def get_dataset_path():
    return "../vgg_data/horse2zebra/"

            
if __name__ == "__main__":

  # preprocess and load data
    print("-----------Preprocess and Load data:---------------")
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                              transforms.Resize((pretrained_size, pretrained_size)),
                              transforms.RandomRotation(5),
                              transforms.RandomHorizontalFlip(0.5),
                              transforms.RandomCrop(pretrained_size, padding=10),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=pretrained_means,
                                                    std=pretrained_stds)
                          ])

    test_transforms = transforms.Compose([
                              transforms.Resize(pretrained_size),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=pretrained_means,
                                                    std=pretrained_stds)
                          ])
    ROOT = get_dataset_path() 
    

    train_data = datasets.ImageFolder(ROOT+"train", transform=train_transforms)

    test_data = datasets.ImageFolder(ROOT+"test", transform=test_transforms)

    VALID_RATIO = 0.7

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples
    

    train_data, valid_data = data.random_split(train_data,
                                           [n_train_examples, n_valid_examples])  

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')   

    BATCH_SIZE = 128

    train_iterator = data.DataLoader(train_data,
                                    shuffle=True,
                                    batch_size=BATCH_SIZE)

    valid_iterator = data.DataLoader(valid_data,
                                    batch_size=BATCH_SIZE)

    test_iterator = data.DataLoader(test_data,
                                    batch_size=BATCH_SIZE)


    # load VGG16 model
    print("-----------Load VGG19 model:---------------")
    OUTPUT_DIM = 2
    vgg19_layers = get_vgg_layers(vgg19_config, batch_norm=True)
    model = VGG(vgg19_layers, OUTPUT_DIM)
    model.load_state_dict(torch.load('vgg19_3_4_23.pt', map_location='cpu'))
    model.eval()
    #print(model)

    # Set training parameters
    print("-----------Set training parameters:---------------")
    START_LR = 1e-7

    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    print("parameters:", optimizer, device, criterion)

    # Load model to device
    print("-----------Load model to device:---------------")
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = model.to(device)
    criterion = criterion.to(device)

    # training
    print("-----------Training:---------------")
    EPOCHS = 100

    best_valid_loss = float('inf')

    for epoch in trange(EPOCHS, desc="Epochs"):
        #print("Inside for loop")
        start_time = time.monotonic()

        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        #train(model, train_iterator, optimizer, criterion, device)
        
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'vgg19_3_4_23.pt')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        #print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    #print("Outside for loop")


