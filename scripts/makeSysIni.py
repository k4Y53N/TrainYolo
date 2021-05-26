import configparser
from pathlib import Path

config = configparser.ConfigParser()
ckpt = Path('./checkpoints')
data_path = Path('./data')

config['Paths'] = {
    'train_set_dir': data_path / 'train2014',
    'train_annotation_path': data_path / 'instances_train2014.json',
    'val_set_dir': data_path / 'val2014',
    'val_annotation_path': data_path / 'instances_val2014.json',
}
config['Save_dir'] = {
    'checkpoints': ckpt,
    'weights': ckpt / 'checkpoints' / 'weights',
    'train_processed_data': data_path / 'Sets' / 'train',
    'test_processed_data': data_path / 'Sets' / 'test',
    'yolo_config_path': Path('./configs')
}

with open('../sys.ini', 'w') as f:
    config.write(f)
