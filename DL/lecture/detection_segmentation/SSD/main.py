import os
import random
import time
import cv2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data

import dataset
from model import SSD
from MultiBoxLoss import MultiBoxLoss

import set_utils


def od_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # 이미지
        targets.append(torch.FloatTensor(sample[1]))  # sample[1] 어노테이션 정보

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None: 
            nn.init.constant_(m.bias, 0.0)


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, device):
    model.to(device)
    # 고속화
    # torch.backends.cudnn.benchmark = True

    iteration = 1
    epoch_train_loss = 0.0  
    epoch_val_loss = 0.0  
    logs = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs+1):
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                print('train')
            else:
                if((epoch+1) % 5 == 0):
                    model.eval()
                    print('-------------')
                    print('val')
                else:
                    continue

            for images, targets in dataloaders_dict[phase]:
                images = images.to(device)
                targets = [ann.to(device) for ann in targets] 

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'): # inference일 때, with torch.no_grad와 같은 역할
                    outputs = model(images)

                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == 'train':
                        loss.backward() 

                        nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)

                        optimizer.step()

                        if (iteration % 10 == 0):
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item(), duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    else:
                        epoch_val_loss += loss.item()

        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss, epoch_val_loss))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        train_loss_list.append(epoch_train_loss)
        val_loss_list.append(epoch_val_loss)


        if epoch_val_loss <= val_loss_list[-1]: # if ((epoch+1) % 10 == 0):
            torch.save(model.state_dict(), 'weights' + str(epoch+1) + '.pth')
        
        if epoch % 10 == 0:
            post_msg = f'LR: {0.001}\nEPOCH : {epoch}\ntrain loss : {epoch_train_loss}\nval loss : {epoch_val_loss}'
            res_ = set_utils._post_message(post_msg)
            set_utils.send_plot(train_loss_list, val_loss_list)
        
        epoch_train_loss = 0.0  
        epoch_val_loss = 0.0  

    return train_loss_list, val_loss_list

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # rootpath = './voc_data/'
    rootpath = '/home/seongwoo/workspace/DataScience_ML-DL/DL/lecture/detection_segmentation/SSD/voc_data/'
    train_img_list, train_anno_list, val_img_list, val_anno_list = dataset.make_datapath_list(rootpath)

    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

    color_mean = (104, 117, 123) 
    input_size = 300  
    batch_size = 32
    num_epochs= 50

    train_dataset = dataset.VOCDataset(train_img_list, train_anno_list, phase="train", transform=dataset.DataTransform(input_size, color_mean), 
                                        transform_anno=dataset.Anno_xml2list(voc_classes))

    val_dataset = dataset.VOCDataset(val_img_list, val_anno_list, phase="val", transform=dataset.DataTransform(input_size, color_mean), 
                                        transform_anno=dataset.Anno_xml2list(voc_classes))



    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn, num_workers=6)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn, num_workers=6)

    print('Dataset is ready!')

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}


    ssd_cfg = {
        'num_classes': 21,  
        'input_size': 300, 
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  
        'feature_maps': [38, 19, 10, 5, 3, 1],  
        'steps': [8, 16, 32, 64, 100, 300],  
        'min_sizes': [30, 60, 111, 162, 213, 264], 
        'max_sizes': [60, 111, 162, 213, 264, 315], 
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    model = SSD(phase="train", cfg=ssd_cfg)
    # vgg_weights = torch.load('./weights/vgg16_reducedfc.pth')
    # net.vgg.load_state_dict(vgg_weights)

    model.extras.apply(weights_init)
    model.loc.apply(weights_init)
    model.conf.apply(weights_init)
    print('model is ready')

    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    train_loss_list, val_loss_list = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, device=device)
    post_msg = f'학습이 완료되었습니다\nmin val loss : {min(val_loss_list)}\n ephoc of best val loss : {val_loss_list.index(min(val_loss_list))+1}'
    
    print('Done!')
