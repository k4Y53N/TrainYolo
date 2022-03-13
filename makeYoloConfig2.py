from sklearn.cluster import KMeans
from pathlib import Path
from threading import Thread
from configparser import ConfigParser
import json
import argparse
import numpy as np
import logging as log
import os
import pickle
import sys

"""
[Annotations]
train_set_dir = data/train2014
train_annotation_path = data/instances_train2014.json
test_set_dir = data/val2014
test_annotation_path = data/instances_val2014.json

[Save_dir]
checkpoints_save_dir = checkpoints
weights_save_dir = checkpoints/weights
"""


class MakeYoloConfig:
    def __init__(
            self,
            config_file_name: str,
            classes_file_path: str,
            sys_config_path: str = './sys.ini',
            pretrain_file_path: str = '',
            model_type: str = 'yolov4',
            frame_work: str = 'tf',
            size: int = 416,
            batch_size: int = 4,
            epoch: int = 30,
            train_size: int = 1000,
            val_size: int = 200,
            tiny: bool = False,

    ):
        self.sys_config = ConfigParser()
        self.sys_config_file = Path(sys_config_path)
        self.name = config_file_name
        self.classes_file = Path(classes_file_path)
        self.pretrain_file = Path(pretrain_file_path) if pretrain_file_path else None
        self.model_type = model_type
        self.frame_work = frame_work
        self.size = size
        self.batch_size = batch_size
        self.epoch = epoch
        self.warmup_epochs = 2
        self.first_stage_epochs = int(epoch * 1 / 5) if pretrain_file_path else 0
        self.second_stage_epochs = epoch
        self.train_size = train_size
        self.test_size = val_size
        self.tiny = tiny
        self.yolo_config = {}
        self.classes = []
        self.anchors = []
        self.anchors_v3 = []
        self.anchors_tiny = []
        self.yolo_config_file: Path = Path()
        self.model_save_dir: Path = Path()
        self.train_set_dir: Path = Path()
        self.train_annotation_file: Path = Path()
        self.test_set_dir: Path = Path()
        self.test_annotation_file: Path = Path()
        self.checkpoints_save_dir: Path = Path()
        self.weights_save_dir: Path = Path()
        self.configs_save_dir: Path = Path()
        self.train_bbox_save_dir: Path = Path()
        self.test_bbox_save_dir: Path = Path()
        self.train_bbox_file: Path = Path()
        self.test_bbox_file: Path = Path()

    def make(self):
        self.check_all_paths()
        self.load_classes()
        train = Thread(
            target=self.write,
            args=(self.train_annotation_file, self.train_bbox_file, True)
        )
        test = Thread(
            target=self.write,
            args=(self.test_annotation_file, self.test_bbox_file, False)
        )
        train.start()
        test.start()
        train.join()
        test.join()

        if self.train_bbox_file.is_file() and self.test_bbox_file.is_file():
            self.yolo_config['name'] = self.name
            self.yolo_config['model_path'] = self.model_type
        else:
            self.yolo_config_file.unlink(missing_ok=True)
            self.train_bbox_file.unlink(missing_ok=True)
            self.test_bbox_file.unlink(missing_ok=True)
            raise RuntimeError('Writing Train bbox file or Test box file fail')

        self.write_yolo_config()

    def check_all_paths(self):
        if self.sys_config_file.is_file():
            log.info(f'System config file path: {self.sys_config_file.absolute()}')
            self.sys_config.read(self.sys_config_file.name)
        else:
            raise FileNotFoundError(self.sys_config_file.name)

        self.train_set_dir = Path(self.sys_config['Annotations']['train_set_dir'])
        self.train_annotation_file = Path(self.sys_config['Annotations']['train_annotation_path'])
        self.test_set_dir = Path(self.sys_config['Annotations']['test_set_dir'])
        self.test_annotation_file = Path(self.sys_config['Annotations']['test_annotation_path'])
        self.checkpoints_save_dir = Path(self.sys_config['Save_dir']['checkpoints'])
        self.weights_save_dir = Path(self.sys_config['Save_dir']['weights'])
        self.configs_save_dir = Path(self.sys_config['Save_dir']['configs'])
        self.yolo_config_file = self.configs_save_dir / Path(self.name).with_suffix('.json')
        self.model_save_dir = self.checkpoints_save_dir / self.name
        self.train_bbox_save_dir = Path(self.sys_config['Save_dir']['train_processed_data'])
        self.test_bbox_save_dir = Path(self.sys_config['Save_dir']['test_processed_data'])
        self.train_bbox_file = self.train_bbox_save_dir / Path(self.name).with_suffix('.bbox')
        self.test_bbox_file = self.test_bbox_save_dir / Path(self.name).with_suffix('.bbox')

        checking_exist_dir_group = (
            self.train_set_dir,
            self.test_set_dir,
        )

        checking_exist_file_group = (
            self.classes_file,
            self.train_annotation_file,
            self.test_annotation_file,
        )

        make_dirs = (
            self.checkpoints_save_dir,
            self.weights_save_dir,
            self.train_bbox_save_dir,
            self.test_bbox_save_dir,
        )

        for ex_dir in checking_exist_dir_group:
            if not ex_dir.is_dir():
                raise NotADirectoryError(ex_dir.absolute())

        for ex_file in checking_exist_file_group:
            if not ex_file.is_file():
                raise FileNotFoundError(str(ex_file.absolute()))

        for mk_dir in make_dirs:
            if not mk_dir.is_dir():
                os.makedirs(mk_dir.absolute(), exist_ok=True)

        if self.pretrain_file:
            if not self.pretrain_file.is_file():
                raise FileNotFoundError(str(self.pretrain_file.absolute()))

    def load_classes(self):
        with self.classes_file.open('r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def write(self, annotation_file: Path, bbox_file: Path, training: bool):
        try:
            self.write_coco2yolo(annotation_file, bbox_file, training)
        except Exception as e:
            log.error(f'{e.__class__.__name__}({e.args})', exc_info=True)
            if training:
                self.train_bbox_file.unlink(missing_ok=True)
            else:
                self.test_bbox_file.unlink(missing_ok=True)

    def write_coco2yolo(self, annotation_file: Path, bbox_file: Path, training: bool):
        images, classes = self.load_annotation_file(annotation_file)
        classes = {
            class_name: index
            for index, class_name in enumerate(classes)
        }
        done = 0
        anchor_boxes = []

        if training:
            data_save_dir = self.train_set_dir
            set_size = self.train_size
        else:
            data_save_dir = self.test_set_dir
            set_size = self.test_size

        with bbox_file.open('w') as bf:
            for image in images:
                image_file = data_save_dir / image['file_name']

                if done >= set_size:
                    break
                if not image_file.is_file() or len(image['items']) < 1:
                    continue

                bf.write(str(image_file.absolute()) + ' ')

                for item in image['items']:
                    x_min = int(item['bbox'][0])
                    y_min = int(item['bbox'][1])
                    x_max = x_min + int(item['bbox'][2])
                    y_max = y_min + int(item['bbox'][3])
                    anchor_boxes.append(
                        (item['bbox'][2],
                         item['bbox'][3],
                         image['width'],
                         image['height'])
                    )

                    x_min, y_min, x_max, y_max = str(x_min), str(y_min), str(x_max), str(y_max)
                    label = str(classes[item['category_name']])
                    bf.write(','.join([x_min, y_min, x_max, y_max, label]) + ' ')

                bf.write('\n')
                done += 1

        if done < 1:
            raise RuntimeError('bbox file did not have any images')
        log.info(f'{bbox_file.name} have {done} images')

        if training:
            self.calculate_anchor_box(anchor_boxes)
            self.classes = list(classes.keys())

    def load_annotation_file(self, file: Path):
        if file.suffix == '.pickle':
            with file.open('rb') as f:
                log.info(f'Start loading annotation file: {file.absolute()}')
                data = pickle.load(f)
                np.random.shuffle(data['images'])
                np.random.shuffle(data['annotations'])
                log.info(f'loading annotation file: {file.absolute()} finish')
        elif file.suffix == '.json':
            with file.open('r') as f:
                log.info(f'Start loading annotation file: {file.absolute()}')
                data = json.load(f)
                np.random.shuffle(data['images'])
                np.random.shuffle(data['annotations'])
                log.info(f'loading annotation file: {file.absolute()} finish')
        else:
            raise RuntimeError('Unknown file suffix')

        return self._filter(data)

    def _filter(self, data):
        cats = {
            cat['id']: cat['name']
            for cat in data['categories'] if cat['name'] in self.classes
        }

        if len(cats) < 1:
            raise RuntimeError('Can not find any classes in annotation file')

        annos = (
            anno for anno in data['annotations']
            if anno['category_id'] in cats.keys()
        )

        images = {
            img['id']: {
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height'],
                'items': []
            }
            for img in data['images']
        }

        for anno in annos:
            image_id = anno['image_id']
            images[image_id]['items'].append(
                {
                    'bbox': anno['bbox'],
                    'category_name': cats[anno['category_id']]
                }
            )

        images = (
            img for img in images.values() if len(img['items']) > 0
        )

        cats = (class_name for class_name in cats.values())

        return images, cats

    def calculate_anchor_box(self, bbox_wh_img_size):
        w_h = tuple(
            (
                box[0] / box[2] * self.size,
                box[1] / box[3] * self.size,
            )
            for box in bbox_wh_img_size
        )

        self.anchors_tiny = self.calculate_kmeans(w_h, 6)
        self.anchors = self.calculate_kmeans(w_h, 9)
        self.anchors_v3 = self.anchors

    @staticmethod
    def calculate_kmeans(w_h, k):
        x = np.array(w_h)
        kmeans = KMeans(n_clusters=k).fit(x)
        cluster = kmeans.cluster_centers_
        boxes_size = [box[0] * box[1] for box in cluster]
        sort_args = np.argsort(boxes_size)
        arranged_cluster = np.zeros_like(cluster, int)

        for i, arg in enumerate(sort_args):
            arranged_cluster[i] = cluster[arg]

        return arranged_cluster.reshape(-1).tolist()

    def write_yolo_config(self):
        self.yolo_config = {
            'name': self.name,
            'model_path': str(self.model_save_dir / self.name),
            'weight_path': str((self.weights_save_dir / self.name).with_suffix('.h5')),
            'frame_work': str(self.frame_work),
            'model_type': str(self.model_type),
            'size': self.size,
            'tiny': self.tiny,
            'max_output_size_per_class': 40,
            'max_total_size': 50,
            'iou_threshold': 0.5,
            'score_threshold': 0.25 if self.tiny else 0.5,
            'YOLO': {
                'CLASSES': self.classes,
                'ANCHORS': self.anchors,
                'ANCHORS_V3': self.anchors_v3,
                'ANCHORS_TINY': self.anchors_tiny,
                'STRIDES': [8, 16, 32],
                'STRIDES_TINY': [16, 32],
                'XYSCALE': [1.2, 1.1, 1.05],
                'XYSCALE_TINY': [1.05, 1.05],
                'ANCHOR_PER_SCALE': 3,
                'IOU_LOSS_THRESH': 0.5,
            },
            'TRAIN': {
                'ANNOT_PATH': str(self.train_bbox_file),
                'BATCH_SIZE': self.batch_size,
                'INPUT_SIZE': self.size,
                'DATA_AUG': True,
                'LR_INIT': 1e-03,
                'LR_END': 1e-06,
                'WARMUP_EPOCHS': self.warmup_epochs,
                'INIT_EPOCH': 0,
                'FIRST_STAGE_EPOCHS': self.first_stage_epochs,
                'SECOND_STAGE_EPOCHS': self.second_stage_epochs,
                'PRETRAIN': str(self.pretrain_file) if self.pretrain_file else None,
            },
            'TEST': {
                'ANNOT_PATH': str(self.test_bbox_file),
                'BATCH_SIZE': self.batch_size,
                'INPUT_SIZE': self.size,
                'DATA_AUG': False,
                'SCORE_THRESHOLD': 0.25 if self.tiny else 0.5,
                'IOU_THRESHOLD': 0.5,
            }
        }

        with self.yolo_config_file.open('w') as f:
            json.dump(self.yolo_config, f)

        if sys.platform.startswith('linux'):
            cmd = 'python3 -m json.tool %s' % (str(self.yolo_config_file.absolute()))
        else:
            cmd = 'python -m json.tool %s' % (str(self.yolo_config_file.absolute()))
        os.system(cmd)
        log.info(f'Write YOLO Config to {self.yolo_config_file.absolute()}')


def main(args):
    try:
        myc = MakeYoloConfig(
            args.name,
            args.classes,
            sys_config_path='./sys.ini',
            size=args.size,
            model_type=args.model,
            frame_work=args.frame_work,
            tiny=args.tiny,
            pretrain_file_path=args.pretrain,
            batch_size=args.batch_size,
            epoch=args.epoch,
            train_size=args.train_size,
            val_size=args.val_size
        )
        myc.make()
    except Exception as E:
        log.error(f'{E.__class__.__name__}({E.args})', exc_info=True)


if __name__ == '__main__':
    log.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=log.INFO
    )

    parser = argparse.ArgumentParser(description='Make YOLO config file')
    parser.add_argument('-n', '--name', required=False, type=str, help='Config file and model name')
    parser.add_argument('-c', '--classes', required=True, type=str, help='Classes file path')
    parser.add_argument('-s', '--size', type=int, default=416,
                        choices=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
                        help='Image detect size')
    parser.add_argument('-m', '--model', type=str, default='yolov4', choices=['yolov4', 'yolov3'], help='Model type')
    parser.add_argument('-f', '--frame_work', type=str, default='tf', choices=['tf', 'trt', 'tflite'],
                        help='Frame work type')
    parser.add_argument('-t', '--tiny', type=bool, default=False, help='Tiny model?')
    parser.add_argument('-p', '--pretrain', type=str, default='', help='Pretrain weight path')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-ep', '--epoch', type=int, default=30, help='Total of epoch')
    parser.add_argument('-ts', '--train_size', type=int, default=1000, help='Train epoch size')
    parser.add_argument('-vs', '--val_size', type=int, default=200, help='Val epoch size')
    args = parser.parse_args()
    main(args)
    # sys_config_path = 'sys.conf'
    # name = 'test3'
    # cls = 'data/classes/person.txt'
    # mk = MakeYoloConfig(name, cls, sys_config_path)
    # mk.make()
    # pprint(mk.__dict__)
