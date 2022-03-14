from configparser import ConfigParser
from pathlib import Path

default_save_dir = {
    'checkpoints': str(Path('./checkpoints')),
    'weights': str(Path('./weights')),
    'configs': str(Path('./configs')),
    'logs': str(Path('./logs')),
    'train_processed_data': str(Path('./data') / 'bbox' / 'train'),
    'test_processed_data': str(Path('./data') / 'bbox' / 'test'),
}

cfg = ConfigParser()
cfg['Annotations'] = {
    'Train_set_dir': '',
    'train_annotation_path': '',
    'test_set_dir': '',
    'test_annotation_path': '',
}
cfg['Save_dir'] = default_save_dir

with open('sys.ini', 'w') as f:
    cfg.write(f)
