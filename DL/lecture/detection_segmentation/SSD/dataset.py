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
