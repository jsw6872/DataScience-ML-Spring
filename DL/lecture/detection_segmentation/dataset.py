import os
import random
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import os
import random
import sys


class Yolo_dataset(data.Dataset):
    def __init__(self, label_path, train, cfg=None):
        super(Yolo_dataset, self).__init__()
        self.cfg = cfg
        self.train = train
        self.root_path = label_path
        self.boxes = 60
        truth = {}
        if self.train:
            self.label_path = os.path.join(label_path, 'train.txt')
        else:
            self.label_path = os.path.join(label_path, 'val.txt')
        f = open(self.label_path, 'r', encoding='utf-8')
        for line in f.readlines():
            data = line.split(" ")
            truth[data[0]] = []
            for i in data[1:]:
                truth[data[0]].append([int(float(j)) for j in i.split(',')])

        self.truth = truth
        self.imgs = list(self.truth.keys())


    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        img_path = self.imgs[index]

        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)
        img = cv2.imread(os.path.join(self.root_path,img_path[7:]))

        img = cv2.resize(img,(608,608))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        num_objs = len(bboxes_with_cls_id)
        if self.train:
            out_bboxes = []
            # target = bboxes_with_cls_id
            boxes = bboxes_with_cls_id[...,:4]
            # boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]
            # target.append(torch.as_tensor(boxes, dtype=torch.float32))
            label = bboxes_with_cls_id[...,-1]
            boxes = np.hstack((boxes, [label]))
            out_bboxes.append(boxes)
            out_bboxes = np.concatenate(out_bboxes, axis=0)
            out_bboxes1 = np.zeros([self.boxes, 5])
            out_bboxes1[:min(out_bboxes.shape[0], self.boxes)] = out_bboxes[:min(out_bboxes.shape[0], self.boxes)]
        else:
            out_bboxes1 = {}
            # boxes to coco format
            boxes = bboxes_with_cls_id[...,:4]
            boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
            out_bboxes1['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            out_bboxes1['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
            out_bboxes1['image_id'] = torch.tensor([get_image_id(img_path)])
            out_bboxes1['area'] = (out_bboxes1['boxes'][:,3])*(out_bboxes1['boxes'][:,2])
            out_bboxes1['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, out_bboxes1

def get_image_id(filename:str) -> int:
    file_name = filename.split('/')[-1]
    id = int(file_name.split('_')[0])
    return id