import os
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter


dataset_dir = '/Users/jeongseungmin/Desktop/Study/codeStudy/unzipped_data/dataset'
train_json_path = os.path.join(dataset_dir, 'train.json')

# coco 데이터 형식의 train.json load
with open(train_json_path, 'r') as f:
    coco_data = json.load(f)


images_to_annotations = {}
images_to_category = {}

for anno in coco_data['annotations']:
    image_id = anno['image_id']
    # 만약에 images_to_annotaions에 아직 들어가 있지 않다면 []를 추가해서 데이터를 넣을준비를 한다
    if image_id not in images_to_annotations:
        images_to_annotations[image_id] = []
    # image_id 위치에 맞게 해당 annotation을 추가합니다.
    images_to_annotations[image_id].append(anno)
    
    # 추가적으로 해당 이미지의 annotation에 클래스 index를 추가해줍니다.
    if image_id not in images_to_category:
        images_to_category[image_id] = []
    images_to_category[image_id].append(anno['category_id'])


#이미지 리스트 및 해당하는 클래스 라벨 추출
image_ids = list(images_to_annotations.keys())

# 해당 데이터셋의 경우 object detection이라 단순히가 가장 먼저 나오는 class를 해당 class로 선정하였습니다.
# 만일 clasffication인 경우 [0]을 제거 하면 됩니다.
image_labels = [images_to_category[image_id][0] for image_id in image_ids]

# # 예를들어서 3개만 직접 확인해본다 image_id = 0,1,2 안에 각각 어떤 카테고리들이 들어있는지
# for i in range(3):
#     print(image_labels[i])

n_split = 5
skf = StratifiedKFold(n_splits=n_split, shuffle= True, random_state=42)

fold_idx = 0

# StratifiedKFold를 사용하여 데이터를 분할 (각 폴드에서 클래스 분포가 비슷하도록 나누기)
for idx, (train_idx, val_idx) in enumerate(skf.split(image_ids, image_labels)):
    if idx == fold_idx:
        train_image_ids = [image_ids[i] for i in train_idx]
        val_image_ids = [image_ids[i] for i in val_idx]
        break