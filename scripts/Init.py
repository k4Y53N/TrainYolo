import configparser
from pathlib import Path
import os

config = configparser.ConfigParser()
root = Path('../')
ckpt = Path('./checkpoints')
data_path = Path('./data')

weights = ckpt / 'weights'
train_processed_data = data_path / 'Sets' / 'train'
test_processed_data = data_path / 'Sets' / 'test'
train_box_file = data_path / 'Sets' / 'bbox'
yolo_config_path = Path('./configs')

config['Annotations'] = {
    'train_set_dir': data_path / 'train2014',
    'train_annotation_path': data_path / 'instances_train2014.json',
    'val_set_dir': data_path / 'val2014',
    'val_annotation_path': data_path / 'instances_val2014.json',
}
config['Save_dir'] = {
    'checkpoints': ckpt,
    'weights': ckpt / 'weights',
    'train_processed_data': data_path / 'Sets' / 'train',
    'test_processed_data': data_path / 'Sets' / 'test',
    'train_bbox_file': data_path / 'Sets' / 'bbox',
    'yolo_config_path': Path('./configs')
}

if not (root / weights).is_dir():
    os.makedirs(root / weights)
if not (root / train_processed_data).is_dir():
    os.makedirs(root / train_processed_data)
if not (root / test_processed_data).is_dir():
    os.makedirs(root / test_processed_data)
if not (root / train_box_file).is_dir():
    os.makedirs(root / train_box_file)
if not (root / yolo_config_path).is_dir():
    os.makedirs(root / yolo_config_path)

with (root / 'sys.ini').open('w') as f:
    config.write(f)
