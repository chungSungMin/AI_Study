import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# load json: modify the path to your own ‘train.json’ file
annotation = '/Users/jeongseungmin/Desktop/Study/codeStudy/unzipped_data/dataset/train.json'

with open(annotation) as f: data = json.load(f)


var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
X = np.ones(len(data['annotations']), 1)
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in sgkf.split(X, y, groups):
    print("Train : ", groups[train_idx])
    print(" ", y[train_idx])
    print("Val : ", groups[val_idx])
    print(" ", y[val_idx])


# check distribution
import pandas as pd
from collections import Counter

def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())

    return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)]

distrs = [get_distribution(y)]
index = ['training set']

for fold_ind, (train_idx, val_idx) in enumerate(sgkf.split(X,y, groups)):
    train_y, val_y = y[train_idx], y[val_idx]
    train_gr, val_gr = groups[train_idx], groups[val_idx]

    assert len(set(train_gr) & set(val_gr)) == 0 
    distrs.append(get_distribution(train_y))

    distrs.append(get_distribution(val_y))
    index.append(f'train - fold{fold_ind}')
    index.append(f'val - fold{fold_ind}')

categories = [d['name'] for d in data['categories']]
pd.DataFrame(distrs, index=index, columns = [categories[i] for i in range(np.max(y) + 1)])

next(sgkf.split(X,y, groups))[0]

len(set(train_gr))

train_gr

# import random
# import os
# import shutil

# origin_dataset_dir = '/Users/jeongseungmin/Desktop/Study/codeStudy/unzipped_data/dataset'
# new_dataset_dir = '/Users/jeongseungmin/Desktop/Study/codeStudy/unzipped_data/skfold-pseduo'
# input_json_path = '/Users/jeongseungmin/Desktop/Study/codeStudy/unzipped_data/dataset/train.json' #train.json 파일 경로
# val_ratio = 0.1


# for fold_ind, (train_idx, val_idx) in enumerate(sgkf.split(X,y, groups)):
# #json 파일 불러오기
#     with open(input_json_path, 'r') as json_reader:
#         dataset = json.load(json_reader)

#     images = dataset['images'] # dict에서 (key:images)의 values 불러오기
#     categories = dataset['categories']# dict에서 (key:catagories)의 values 불러오기
#     annotations = dataset['annotations']# dict에서 (key:annotations)의 values 불러오기
    
#     train_gr, val_gr = groups[train_idx], groups[val_idx]

#     # image_ids = [x.get('id') for x in images] # get함수를 통해 dict에서 id값 추출
#     # image_ids.sort() # 정렬
#     # random.shuffle(image_ids) # 인덱스 섞기

#     # num_val = int(len(image_ids) * val_ratio)
#     # num_train = len(image_ids) - num_val

#     image_ids_val, image_ids_train = set(val_gr), set(train_gr)

#     num_train = len(image_ids_train)
#     num_val = len(image_ids_val)

#     #Image_id를 기준으로 train/val 나누기
#     train_images = [x for x in images if x.get('id') in image_ids_train]
#     val_images = [x for x in images if x.get('id') in image_ids_val]
#     train_annotations = [x for x in annotations if x.get('image_id') in image_ids_train]
#     val_annotations = [x for x in annotations if x.get('image_id') in image_ids_val]

#     #file_name 수정
#     for info in val_images:
#         name = info['file_name'].split('/')[1]
#         info['file_name'] = os.path.join('val',name)
        
#     #나눈 정보를 가지고 새로운 dict 생성
#     train_data = {
#         'images': train_images,
#         'annotations': train_annotations,
#         'categories': categories,
#     }

#     val_data = {
#         'images': val_images,
#         'annotations': val_annotations,
#         'categories': categories,
#     }


#     # 새롭게 만든 dict로 train/val json 파일 생성
#     os.makedirs(new_dataset_dir+f'/{fold_ind}', exist_ok=True)

#     new_train_json = os.path.join(new_dataset_dir, f'{fold_ind}','train.json')
#     new_val_json = os.path.join(new_dataset_dir,f'{fold_ind}', 'val.json')
#     copy_test_json = os.path.join(new_dataset_dir, f'{fold_ind}','test.json')

#     #train.json 새롭게 생성
#     with open(new_train_json, 'w') as train_writer:
#         json.dump(train_data, train_writer)

#     #val.json 새롭게 생성
#     with open(new_val_json, 'w') as val_writer:
#         json.dump(val_data, val_writer)

#     # train/val 이미지 파일 분리 복사
#     os.makedirs(os.path.join(new_dataset_dir, f'{fold_ind}','train'), exist_ok=True)
#     os.makedirs(os.path.join(new_dataset_dir, f'{fold_ind}','val'), exist_ok=True)

#     # train 해당 파일 복사
#     for train_img_info in train_images:
#         from_copy_train_img = os.path.join(origin_dataset_dir, train_img_info['file_name'])
#         to_copy_train_img = os.path.join(new_dataset_dir, f'{fold_ind}',train_img_info['file_name'])
#         shutil.copyfile(from_copy_train_img, to_copy_train_img)
        
#     # val 해당 파일 복사
#     for val_img_info in val_images:
#         origin_id = os.path.join('train', val_img_info['file_name'].split('/')[1])
#         from_copy_val_img = os.path.join(origin_dataset_dir, origin_id)
#         to_copy_val_img = os.path.join(new_dataset_dir,f'{fold_ind}', val_img_info['file_name'])
#         shutil.copyfile(from_copy_val_img, to_copy_val_img)
    
#     #기존 파일에서 test json파일 복사
#     shutil.copyfile(os.path.join(origin_dataset_dir, 'test.json'), copy_test_json)

#     # test 이미지 폴더 전체 복사
#     shutil.copytree(os.path.join(origin_dataset_dir, 'test'), os.path.join(new_dataset_dir,f'{fold_ind}', 'test'))


#     print(f'train 이미지 파일 개수({int((1-val_ratio)*100)}%):{num_train}')
#     print('new_dataset_train 파일 개수:{}'.format(len(os.listdir(os.path.join(new_dataset_dir,f'{fold_ind}','train')))))
#     print(f'val 이미지 파일 개수({int(val_ratio*100)}%):{num_val}')
#     print('new_dataset_val 파일 개수:{}'.format(len(os.listdir(os.path.join(new_dataset_dir,f'{fold_ind}', 'val')))))