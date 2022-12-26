import os
import shutil
import glob
import tarfile

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


TARGET_PATH = "./data"
DATA_PATH = './data'
def make_breed_name(TARGET_PATH):
    # 파일 이름 변경
    for dir_name in os.listdir(TARGET_PATH):
        breed_name = '_'.join(dir_name.split('-')[1:]).lower()

        source_dir = os.path.join(TARGET_PATH, dir_name)
        target_dir = os.path.join(TARGET_PATH, breed_name)
        
        shutil.move(source_dir, target_dir) # source_dir파일을 target_dir로 이름 변경


def make_dataset(TARGET_PATH):
    dataset = []

    for filepath in glob.iglob(f'{TARGET_PATH}/**/*.jpg', recursive=True): # TARGET_PATH 기준 모든 하위 디렉토리에서 해당 확장자 파일을 검출하고 싶을 때
        breed_name = filepath.split('/')[2]
        dataset.append([filepath, breed_name])
    dataset = np.array(dataset)



    train_val_image, test_image, train_val_target, test_target = train_test_split(dataset[:,0], dataset[:,1], stratify=dataset[:,1], test_size=0.2)
    train_image, val_image, train_target, val_target = train_test_split(train_val_image, train_val_target, stratify=train_val_target, test_size=0.2)


    if os.path.exists(DATA_PATH):
        # os.mkdir(DATA_PATH) # 단일 폴더만 생성 가능
        os.makedirs(os.path.join(DATA_PATH, 'train')) # 폴더를 여러개 만들기 가능
        os.makedirs(os.path.join(DATA_PATH, 'val'))
        os.makedirs(os.path.join(DATA_PATH, 'test'))

        for breed_name in set(test_target):
            os.makedirs(os.path.join(DATA_PATH, 'train', breed_name))
            os.makedirs(os.path.join(DATA_PATH, 'val', breed_name))
            os.makedirs(os.path.join(DATA_PATH, 'test', breed_name))


    for filepath, target_dir in zip(train_image.tolist(), train_target.tolist()):
        filename = filepath.split('/')[-1]
        source_path = filepath
        target_dir = os.path.join(DATA_PATH, 'train', target_dir, filename)
        print(source_path, '---------',target_dir)
        shutil.move(source_path, target_dir) # shutil.copy(복사할 파일, 복사위치+파일명)

    for filepath, target_dir in zip(val_image.tolist(), val_target.tolist()):
        filename = filepath.split('/')[-1]
        source_path = filepath
        target_dir = os.path.join(DATA_PATH, 'val', target_dir, filename)
        print(source_path, target_dir)
        shutil.move(source_path, target_dir)

    for filepath, target_dir in zip(test_image.tolist(), test_target.tolist()):
        filename = filepath.split('/')[-1]
        source_path = filepath
        target_dir = os.path.join(DATA_PATH, 'test', target_dir, filename)
        print(source_path, target_dir)
        shutil.move(source_path, target_dir)

    for dir_name in os.listdir(DATA_PATH):
        if dir_name not in ['train', 'test', 'val']:
            shutil.rmtree(DATA_PATH+'/'+dir_name)