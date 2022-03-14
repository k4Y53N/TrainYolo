import json
import numpy as np


def load_json(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


class YOLOConfiger:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_json(config_path)
        self.name = self.config['name']
        self.model_path = self.config['model_path']
        self.weight_path = self.config['weight_path']
        self.frame_work = self.config['frame_work']
        self.model_type = self.config['model_type']
        self.size = self.config['size']
        self.tiny = self.config['tiny']
        self.max_output_size_per_class = self.config['max_output_size_per_class']
        self.max_total_size = self.config['max_total_size']
        self.iou_threshold = self.config['iou_threshold']
        self.score_threshold = self.config['score_threshold']
        self.logdir = self.config['logdir']
        self.classes = self.config['YOLO']['CLASSES']
        self.anchor_per_scale = self.config['YOLO']['ANCHOR_PER_SCALE']
        self.iou_loss_thresh = self.config['YOLO']['IOU_LOSS_THRESH']
        self.train_annot_path = self.config['TRAIN']['ANNOT_PATH']
        self.train_batch_size = self.config['TRAIN']['BATCH_SIZE']
        self.lr_init = self.config['TRAIN']['LR_INIT']
        self.lr_end = self.config['TRAIN']['LR_END']
        self.warmup_epochs = self.config['TRAIN']['WARMUP_EPOCHS']
        self.init_epoch = self.config['TRAIN']['INIT_EPOCH']
        self.first_stage_epochs = self.config['TRAIN']['FIRST_STAGE_EPOCHS']
        self.second_stage_epochs = self.config['TRAIN']['SECOND_STAGE_EPOCHS']
        self.pre_train_file_path = self.config['TRAIN']['PRETRAIN']
        self.test_annot_path = self.config['TEST']['ANNOT_PATH']
        self.test_batch_size = self.config['TEST']['BATCH_SIZE']
        self.num_class = len(self.classes)
        self.strides = []
        self.anchors = []
        self.xyscale = []
        self.freeze_layers = []
        if self.tiny:
            self.strides = np.array(self.config['YOLO']['STRIDES_TINY'])
            self.anchors = np.array(self.config['YOLO']['ANCHORS_TINY']).reshape((2, 3, 2))
            if self.model_type == 'yolov4':
                self.xyscale = self.config['YOLO']['XYSCALE_TINY']
                self.freeze_layers = ['conv2d_17', 'conv2d_20']
            else:
                self.xyscale = [1, 1]
                self.freeze_layers = ['conv2d_9', 'conv2d_12']
        else:
            self.strides = np.array(self.config['YOLO']['STRIDES'])
            self.anchors = np.array(self.config['YOLO']['ANCHORS']).reshape((3, 3, 2))
            self.xyscale = self.config['YOLO']['XYSCALE'] if self.config['model_type'] == 'yolov4' else [1, 1, 1]
            if self.model_type == 'yolov4':
                self.xyscale = self.config['YOLO']['XYSCALE']
                self.freeze_layers = ['conv2d_93', 'conv2d_101', 'conv2d_109']
            else:
                self.xyscale = [1, 1, 1]
                self.freeze_layers = ['conv2d_58', 'conv2d_66', 'conv2d_74']

    def update_init_epoch(self, epoch: int):
        self.init_epoch = epoch
        self.config['TRAIN']['INIT_EPOCH'] = epoch

    def update_pre_train_file_path(self, pre_train_file_path: str):
        self.pre_train_file_path = pre_train_file_path
        self.config['TRAIN']['PRETRAIN'] = pre_train_file_path

    def save(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
