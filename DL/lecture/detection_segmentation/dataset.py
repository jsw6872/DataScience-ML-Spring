import os
import random
# XML 다루는 라이브러리
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import os
import random
import sys

from data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

# 학습 및 검증용 화상 데이터, 어노테이션 데이터 파일 경로 리스트 작성
def make_datapath_list(rootpath):
    img_file_path = os.path.join(rootpath, 'JPEGImages', '%s.jpg') # %연산 시 3번째 arg형태로 파일명이 됨
    annotation_path = os.path.join(rootpath, 'Annotations', '%s.xml')

    train_file_id = os.path.join(rootpath + 'ImageSets/Main/train.txt')
    val_file_id = os.path.join(rootpath + 'ImageSets/Main/val.txt')

    train_img_list = []
    train_annotation_list = []

    for line in open(train_file_id): # txt파일을 한줄씩 읽어옴
        file_id = line.strip() 
        img_file_name = (img_file_path % file_id)  # 画像のパス
        anno_file_name = (annotation_path % file_id)  # アノテーションのパス
        train_img_list.append(img_file_name)  # リストに追加
        train_annotation_list.append(anno_file_name)  # リストに追加

    val_img_list = []
    val_annotation_list = []

    for line in open(val_file_id): # txt파일을 한줄씩 읽어옴
        file_id = line.strip()  
        img_file_name = (img_file_path % file_id)  
        anno_file_name = (annotation_path % file_id)  
        val_img_list.append(img_file_name)  # 
        val_annotation_list.append(anno_file_name)  #

    return train_img_list, train_annotation_list, val_img_list, val_annotation_list


# rootpath = './data/VOCdevkit/VOC2012/'
# train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)


# voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
#                'bottle', 'bus', 'car', 'cat', 'chair',
#                'cow', 'diningtable', 'dog', 'horse',
#                'motorbike', 'person', 'pottedplant',
#                'sheep', 'sofa', 'train', 'tvmonitor']


class Anno_xml2list(object): 
    def __init__(self, classes):

        self.classes = classes

    def __call__(self, xml_path, width, height):
        # 물체의 어노테이션 데이터를 저장한 리스트. 이미지에 존재하는 물체 수만큼 len을 가짐.
        ret = [] #ret : [[xmin, ymin, xmax, ymax, label_ind], ... ]

        # xml파일로드
        xml = ET.parse(xml_path).getroot()

        # xml에서 <object>의 수만큼 반복
        for obj in xml.iter('object'):

            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # 한 물체의 바운딩 박스 어노테이션 정보를 저장하는 리스트
            bndbox = []

            name = obj.find('name').text.lower().strip()  # 물체이름
            bbox = obj.find('bndbox')  # 바운딩박스 정보

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # VOC데이터셋은 원점이 (1,1)이므로 빼줌
                cur_pixel = int(bbox.find(pt).text) - 1

                # 정규화
                if pt == 'xmin' or pt == 'xmax':  # x의 경우 width로 나눔
                    cur_pixel /= width
                else:  # y는 높이인 height로 나눔
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            # 어노테이션의 class명에 해당하는 index번호 추가
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            ret += [bndbox]

        return np.array(ret) #[[xmin, ymin, xmax, ymax, label_ind], ... ]

# transform_anno = Anno_xml2list(voc_classes)

# ind = 1
# image_file_path = val_img_list[ind]
# img = cv2.imread(image_file_path)  # [높이][폭][RGB]
# height, width, channels = img.shape  # 이미지 크기


# 이미지와 bbox전처리 실시
class DataTransform():
    def __init__(self, input_size, color_mean): # color_mean : (BGR) - cv2는 BGR순서
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),  # 
                ToAbsoluteCoords(),  # 
                PhotometricDistort(),  # 
                ToPercentCoords(),  # 
                Resize(input_size),  # 
                SubtractMeans(color_mean)  # 
            ]),
            'val': Compose([
                ConvertFromInts(),  # 
                Resize(input_size),  # 
                SubtractMeans(color_mean)  
            ])
        }

    def __call__(self, img, phase, boxes, labels): # phase : 'train' or 'val'

        return self.data_transform[phase](img, boxes, labels)



# # 이미지 불러오기
# image_file_path = train_img_list[0]
# img = cv2.imread(image_file_path)  # [높이][폭][BGR]
# height, width, channels = img.shape  # 이미지 shape

# # 어노테이션을 리스트로
# transform_anno = Anno_xml2list(voc_classes)
# anno_list = transform_anno(train_anno_list[0], width, height)

# # 원래 이미지 표시
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# # data 변형
# color_mean = (104, 117, 123)  # (BGR) 색상의 평균값
# input_size = 300  # 이미지input 사이즈를 300x300 으로
# transform_data = DataTransform(input_size, color_mean)

# # 5. train 변형
# phase = "train"
# img_transformed, boxes, labels = transform_data(
#     img, phase, anno_list[:, :4], anno_list[:, 4])
# plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
# plt.show()


# # 6. val 변형
# phase = "val"
# img_transformed, boxes, labels = transform_data(
#     img, phase, anno_list[:, :4], anno_list[:, 4])
# plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
# plt.show()



class VOCDataset(data.Dataset):
    """
    torch.utils.data.Dataset 상속받음
    """

    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list # 이미지 경로 저장된 리스트
        self.anno_list = anno_list # 어노테이션 경로 저장된 리스트
        self.phase = phase  # train or val 지정
        self.transform = transform  # 이미지 변경
        self.transform_anno = transform_anno  # xml 정보 리스트 변경

    def __len__(self):
        return len(self.img_list) # 이미지 리스트 개수 반환

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  # [높이][색][BGR]
        height, width, channels = img.shape  # [높이][색][BGR]

        anno_file_path = self.anno_list[index] 
        anno_list = self.transform_anno(anno_file_path, width, height) # 해당 높이와 너비로 anno_list 만들기

        img, boxes, labels = self.transform(img, self.phase, anno_list[:, :4], anno_list[:, 4]) # BGR -> RGB, (높이, 폭, 채널) -> (채널, 높이, 폭)
 
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
 
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1))) # bbox와 라벨을 한 세트로

        return img, gt, height, width



# color_mean = (104, 117, 123)  # (BGR) 평균값
# input_size = 300  # 이미지 input size

# train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
#     input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

# val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
#     input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))


# # 데이터 출력 예(image, bbox+label)
# a,b = val_dataset[11] #== val_dataset.__getitem__(1)




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

        # if self.train:
        #     self.dataset_dir = f'{cfg.dataset_dir}/train'
        # else:
        #     self.dataset_dir = f'{cfg.dataset_dir}/valid'

    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
    # def _get_val_item(self, index):
        img_path = self.imgs[index]
        # if self.train:
        #     img_path = random.choice(list(self.truth.keys()))

        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)
        img = cv2.imread(os.path.join(self.root_path,img_path[7:]))
        # img = cv2.imread(os.path.join(self.cfg.dataset_dir, img_path))
        # img_height, img_width = img.shape[:2]
        img = cv2.resize(img,(608,608))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.cfg.w, self.cfg.h))
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        num_objs = len(bboxes_with_cls_id)
        if self.train:
            out_bboxes = []
            # target = bboxes_with_cls_id
            boxes = bboxes_with_cls_id[...,:4]
            boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]
            # target.append(torch.as_tensor(boxes, dtype=torch.float32))
            label = bboxes_with_cls_id[...,-1]
            boxes = np.hstack((boxes, [label]))
            out_bboxes.append(boxes)
            out_bboxes = np.concatenate(out_bboxes, axis=0)
            # out_bboxes1 = np.zeros([self.boxes, 5])
            # out_bboxes1[:min(out_bboxes.shape[0], self.boxes)] = out_bboxes[:min(out_bboxes.shape[0], self.boxes)]
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
        return img, out_bboxes#out_bboxes1

def get_image_id(filename:str) -> int:
    # raise NotImplementedError("Create your own 'get_image_id' function")
    # lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    # lv = lv.replace("level", "")
    # no = f"{int(no):04d}"
    # return int(lv+no)
    file_name = filename.split('/')[-1]
    id = int(file_name.split('_')[0])
    # print("You could also create your own 'get_image_id' function.")
    # print(filename)
    # parts = filename.split('/')
    # id = int(parts[-1][0:-4])
    # print(id)
    return id