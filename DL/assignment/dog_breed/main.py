import numpy as np
import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import torchsummary
from torch.optim.lr_scheduler import StepLR 

import sys
import tempfile
import time
import copy

import model
import set_utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

EPOCHS = 400
BATCH_SIZE = 64
LEARNING_RATE = 0.01


def train(model, train_loader, optimizer):
    model.train()  
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad() 
        output = model(data)  
        loss = F.cross_entropy(output, target).cuda()
        loss.backward()  
        optimizer.step()


def train_baseline(model ,train_loader, val_loader, optimizer, num_epochs):
    train_loss_list = []
    val_loss_list = []

    best_acc = 0.0  
    best_model_wts = copy.deepcopy(model.state_dict()) 
 
    for epoch in range(1, num_epochs + 1):
        since = time.time()  
        train(model, train_loader, optimizer)
        train_loss, train_acc = evaluate(model, train_loader) 
        val_loss, val_acc = evaluate(model, val_loader)
        
        if val_acc > best_acc: 
            best_acc = val_acc 
            best_model_wts = copy.deepcopy(model.state_dict())
        
        time_elapsed = time.time() - since 
        print('-------------- epoch {} ----------------'.format(epoch))
        print('train Loss: {:.4f}, train Accuracy: {:.2f}%'.format(train_loss, train_acc))   
        print('val Loss: {:.4f}, val Accuracy: {:.2f}%'.format(val_loss, val_acc))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        train_loss_list.append(round(train_loss, 5))
        val_loss_list.append(round(val_loss, 5))

        if epoch % 40 == 0:
            post_msg = f'LR: {LEARNING_RATE}\nEPOCH : {epoch}\ntrain loss : {train_loss}\nval loss : {val_loss}\nbest val acc : {best_acc}'
            res_ = set_utils._post_message(post_msg)
            set_utils.send_plot(train_loss_list, val_loss_list)

    model.load_state_dict(best_model_wts)  
    return model, train_loss_list, val_loss_list


def evaluate(model, test_loader):
    model.eval()  
    test_loss = 0 
    correct = 0   
    
    with torch.no_grad(): 
        for data, target in test_loader:  
            data, target = data.to(device), target.to(device)  
            output = model(data) 
            
            test_loss += F.cross_entropy(output,target, reduction='sum').item() 
 
            
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item() 
   
    test_loss /= len(test_loader.dataset) 
    test_accuracy = 100. * correct / len(test_loader.dataset) 
    return test_loss, test_accuracy  


def main():

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       # transforms.RandomVerticalFlip(),
                                       # transforms.RandomAffine(180, shear=20),
                                       transforms.ToTensor(), # 정규화 결과가 0 ~ 1
                                    #    transforms.Resize((224, 224)),
                                       transforms.RandomErasing(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                     ])

                                    #   transforms.CenterCrop(224),
    val_transforms = transforms.Compose([transforms.Resize(225),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                        ])

    test_transforms = transforms.Compose([transforms.Resize(225),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                                        ])

    DATASET_PATH = './data'
    train_data = datasets.ImageFolder(DATASET_PATH + '/train', transform=train_transforms)
    val_data = datasets.ImageFolder(DATASET_PATH + '/val', transform=val_transforms)
    test_data = datasets.ImageFolder(DATASET_PATH + '/test', transform=test_transforms)

    train_iter = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    val_iter = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)                           
    
    train_labels_map = {v:k for k, v in train_data.class_to_idx.items()}
    
    model = model.DenseNet_121(len(train_labels_map)).to(device)

    # criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.95, weight_decay=0.001)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    try:
        base, train_loss_list, val_loss_list = train_baseline(model, train_iter, val_iter, optimizer, EPOCHS)
        torch.save(base,'baseline.pt')
        
        train_loss_msg = str(train_loss_list[-5])
        # val_loss_msg = str(val_loss_list[-5])
        
        for i in range(-4, 0):
            train_loss_msg += f' ---> {train_loss_list[i]}'
        
        post_msg = f'학습이 완료되었습니다\ntrain loss : {train_loss_msg}\nmin val loss : {min(val_loss_list)}\n ephoc of best val loss : {val_loss_list.index(min(val_loss_list))}'
        res_ = set_utils._post_message(post_msg)


    except Exception as e:
        res_ = set_utils._post_message(f'Error : {e}')
        # print(e)
    return True

if __name__ == '__main__':
    main()