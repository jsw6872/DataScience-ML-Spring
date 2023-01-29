import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import utils
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import transforms as T


class COCO_dataformat(Dataset):
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform # transform : albumentations 라이브러리 사용
        self.coco = COCO(data_dir)

        self.cat_ids = self.coco.getCatIds() # category id 반환
        self.cats = self.coco.loadCats(self.cat_ids) # category id를 입력으로 category name, super category 정보 담긴 dict 반환
        self.classNameList = ['Backgroud'] # class name 저장 
        for i in range(len(self.cat_ids)):
          self.classNameList.append(self.cats[i]['name'])

    def __getitem__(self, index):
        image_id = self.coco.getImgIds(imgIds=index) # img id 또는 category id 를 받아서 img id 반환
        image_infos = self.coco.loadImgs(image_id)[0] # img id를 받아서 image info 반환
        
        # cv2 를 활용하여 image 불러오기(BGR -> RGB 변환 -> numpy array 변환 -> normalize(0~1))
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0 # albumentations 라이브러리로 toTensor 사용시 normalize 안해줘서 미리 해줘야~
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id']) #img id, category id를 받아서 해당하는 annotation id 반환
            anns = self.coco.loadAnns(ann_ids) # annotation id를 받아서 annotation 정보 반환

            # 저장된 annotation 정보로 label mask 생성, Background = 0, 각 pixel 값에는 "category id" 할당
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=False)
            for i in range(len(anns)): # 이미지 하나에 존재하는 annotation 순회
                pixel_value = anns[i]['category_id'] # 해당 클래스 이름의 인덱스
                #className = classNameList[anns[i]['category_id']] # 클래스 이름
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value # coco.annToMask(anns) : anns 정보로 mask를 생성 / 객체가 있는 곳마다 객체의 label에 해당하는 mask 생성
            masks = masks.astype(np.int8)
            
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

            # if self.transform is not None:
            #     transformed = self.transform(image=images, mask=masks)
            #     images = transformed["image"]
            #     masks = transformed["mask"]
            # return images, masks, image_infos
        
        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        return len(self.coco.getImgIds()) # 전체 dataset의 size 반환 


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == '__main__':
    dataset_path = '/home/seongwoo/workspace/DataScience_ML-DL/DL/assignment/detection/data' # Dataset 경로 지정 필요
    train_path = dataset_path + '/train.json'
    dataset = COCO_dataformat(data_dir=train_path, transform=get_transform(train=True))

    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,
                                                    collate_fn=utils.collate_fn)

    batch = iter(train_data_loader)
    data = next(batch)