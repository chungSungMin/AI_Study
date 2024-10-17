from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os 

data_dir = '../../lv1_data/data'
data = pd.read_csv(os.path.join(data_dir, 'train.csv'))

target = data['target']
image_path  = data['image_path']


skf = StratifiedKFold(n_splits=5, random_state= 42, shuffle= True)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(image_path, target)):
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]

    print(f'{fold_idx}번쨰 데이터 생성되었습니다.')
    print(f'train_data의 개수 : {len(train_data)}')
    print(f'val_data의 개수 : {len(val_data)}')


    train_data.to_csv(f'{data_dir}/train_fold_{fold_idx}.csv', index = False)
    val_data.to_csv(f'{data_dir}/val_fold_{fold_idx}.csv', index = False )

    print(f"Fold {fold_idx}: train and validation data saved.\n")