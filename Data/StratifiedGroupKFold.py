import json
import os
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# 데이터 경로 설정
data_dir = '../../dataset'
train_json_path = os.path.join(data_dir, "train.json")

# coco 데이터 로드
with open(train_json_path, 'r') as f:
    coco_data = json.load(f)

'''
    image_id와 category_id를 이용해 X, y, group 생성
    X : 형식상 필요하면 stratifiedGroupKFold에서는 필요로 하지 않습니다.
    y : 실제 레이블 ( 해당 task 에서는 category_id ) 를 설정
    group : 그룹을 만들어줍니다 ( 하나의 이미지내의 모든 객체들은 동일한 그룹에 들어가야 하기에 image_id 가 그룹이 된다)
'''
var = [(anno['image_id'], anno['category_id']) for anno in coco_data['annotations']]
X = np.ones((len(var), 1))  # placeholder, feature 없이 진행하는 경우
y = np.array([an[1] for an in var])
group = np.array([an[0] for an in var])


n_splits = 5
# StratifiedGroupKFold로 데이터 분리
stgk = StratifiedGroupKFold(n_splits= n_splits, shuffle=True, random_state=42)

# fold별로 데이터 저장
for fold_index, (train_idx, val_idx) in enumerate(stgk.split(X, y, group)):
    # train, val 데이터 분리
    train_annotations = [coco_data['annotations'][i] for i in train_idx]
    val_annotations = [coco_data['annotations'][i] for i in val_idx]
    
    # train, val JSON 생성
    train_data = coco_data.copy()
    val_data = coco_data.copy()
    
    # annotations 업데이트
    train_data['annotations'] = train_annotations
    val_data['annotations'] = val_annotations
    
    # 이미지 ID들 추출
    train_image_ids = set(anno['image_id'] for anno in train_annotations)
    val_image_ids = set(anno['image_id'] for anno in val_annotations)
    
    # 이미지 정보 업데이트
    train_data['images'] = [img for img in coco_data['images'] if img['id'] in train_image_ids]
    val_data['images'] = [img for img in coco_data['images'] if img['id'] in val_image_ids]
    
    # train, val JSON 파일로 저장
    fold_train_json_path = os.path.join(data_dir, f"train_fold_{fold_index}.json")
    fold_val_json_path = os.path.join(data_dir, f"val_fold_{fold_index}.json")
    
    # json.dump(올리고자 하는 데이터 값, 올리고 싶은 경로)
    with open(fold_train_json_path, 'w') as f_train:
        json.dump(train_data, f_train)
    
    with open(fold_val_json_path, 'w') as f_val:
        json.dump(val_data, f_val)
    
    print(f"Fold {fold_index}: train and val JSON files saved.")





## |_____________________________________________________________________________________________________________ |
## | 아래 내용의 경우 detectron2에 데이터를 등록하는 방법입니다. detectron2를 사용하는 경우 아래 코드를 실행 시켜 등록을 확인 가능합니다.    |
## |_____________________________________________________________________________________________________________ |



# for fold_idx in range(n_splits):
#     # Register Dataset
#     try:
#         # train_fold_{fold_idx}.json 파일을 등록하는 부분에서 f-string을 사용하여 경로를 올바르게 설정
#         register_coco_instances(f'coco_trash_train_fold_{fold_idx}', {}, 
#                                 f'../../dataset/train_fold_{fold_idx}.json', '../../dataset/')
#     except AssertionError:
#         pass

#     try:
#         # val_fold_{fold_idx}.json 파일을 등록하는 부분에서도 f-string을 사용하여 경로를 올바르게 설정
#         register_coco_instances(f'coco_trash_val_fold_{fold_idx}', {}, 
#                                 f'../../dataset/val_fold_{fold_idx}.json', '../../dataset/')
#     except AssertionError:
#         pass

# # MetadataCatalog는 메타데이터를 설정
# MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
#                                                          "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]


# from detectron2.data import DatasetCatalog

# # 등록된 데이터셋 목록 확인
# print(DatasetCatalog.list())