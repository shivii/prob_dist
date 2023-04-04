import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
import torch.utils.data as data
from tqdm.notebook import tqdm, trange


def get_dataset_path():
        return "../vgg_data/horse2zebra/"

# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

train_iterator = data.DataLoader(train_data,
                                shuffle=True,
                                batch_size=32)
valid_iterator = data.DataLoader(valid_data,
                                batch_size=32)
test_iterator = data.DataLoader(test_data,
                                batch_size=32)
vgg19 = models.vgg19(pretrained=True)
#vgg19.load_state_dict(torch.load("vgg19_23_06.pt"))
#vgg19.eval()
vgg19.to(device)
#print(vgg19)
# change the number of classes 
vgg19.classifier[6].out_features = 10
# freeze convolution weights
for param in vgg19.features.parameters():
    param.requires_grad = False
# optimizer
optimizer = optim.SGD(vgg19.classifier.parameters(), lr=0.001, momentum=0.9)
# loss function
criterion = nn.CrossEntropyLoss()
# validation function
def validate(model, test_dataloader):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for (x, y) in tqdm(test_dataloader, desc="Training", leave=False):
        data, target = x.to(device), y.to(device)
        output = model(data)
        loss = criterion(output, target)
        
        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()
    
    val_loss = val_running_loss/len(test_dataloader.dataset)
    val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
    #print(f'Test Loss: {val_loss:.4f}, Test Acc: {val_accuracy:.2f}')
    return val_loss, val_accuracy
# training function
def fit(model, train_dataloader):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for (x, y) in tqdm(train_dataloader, desc="Training", leave=False):
        data, target = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
    #print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    
    return train_loss, train_accuracy
    
train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []
start = time.time()
best_valid_loss = float('inf')

for epoch in trange(100, desc="Epochs"):
    train_epoch_loss, train_epoch_accuracy = fit(vgg19, train_iterator)
    val_epoch_loss, val_epoch_accuracy = validate(vgg19, valid_iterator)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    
    
    if val_epoch_loss < best_valid_loss:
            best_valid_loss = val_epoch_loss
            torch.save(vgg19.state_dict(), 'vgg19_3_4_23_pytorch.pt')
            print("Training loss, accuracy:", train_epoch_loss, train_epoch_accuracy)
            print("Validation loss, accuracy", val_epoch_loss, val_epoch_accuracy)
            print("saved !!!")
            


end = time.time()
print((end-start)/60, 'minutes')
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validataion accuracy')
plt.legend()
plt.savefig('vgg_accuracy_app2or.png')
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.legend()
plt.savefig('vgg_loss_app2or.png')
plt.show()

# -*- coding: utf-8 -*-

