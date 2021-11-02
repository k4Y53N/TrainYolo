from sklearn.cluster import KMeans
from pathlib import Path
from threading import Thread
from configparser import ConfigParser
from pprint import pprint
import json
import argparse
import numpy as np
import logging
import os
import pickle

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
            sys_config_path: str = './sys.conf ',
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
        self.pretrain_file = Path(pretrain_file_path)
        self.model_type = model_type
        self.frame_work = frame_work
        self.size = size
        self.batch_size = batch_size
        self.epoch = epoch
        self.train_size = train_size
        self.test_size = val_size
        self.tiny = tiny
        self.yolo_config = {}
        self.classes = []
        self.anchors = []
        self.anchors_v3 = []
        self.anchors_tiny = []
        self.yolo_config_file: Path = None
        self.model_save_dir: Path = None
        self.train_set_dir: Path = None
        self.train_annotation_file: Path = None
        self.test_set_dir: Path = None
        self.test_annotation_file: Path = None
        self.checkpoints_save_dir: Path = None
        self.weights_save_dir: Path = None
        self.configs_save_dir: Path = None
        self.train_bbox_save_dir: Path = None
        self.test_bbox_save_dir: Path = None
        self.train_bbox_file: Path = None
        self.test_bbox_file: Path = None

    def make(self):
        self.check_all_paths()
        self.load_classes()
        train = Thread(
            target=self.call_write,
            args=(self.train_annotation_file, self.train_bbox_file, True)
        )
        test = Thread(
            target=self.call_write,
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

    def check_all_paths(self):
        if self.sys_config_file.is_file():
            logging.info(f'System config file path: {self.sys_config_file.name}')
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
        self.yolo_config_file = self.configs_save_dir / Path(self.name).with_suffix('.CONF')
        self.model_save_dir = self.checkpoints_save_dir / Path(self.name)
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
                raise NotADirectoryError(ex_dir.name)

        for ex_file in checking_exist_file_group:
            if not ex_file.is_file():
                raise FileNotFoundError(ex_file.name)

        for mk_dir in make_dirs:
            if not mk_dir.is_dir():
                os.makedirs(mk_dir.name, exist_ok=True)

    def load_classes(self):
        with self.classes_file.open('r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def call_write(self, annotation_file: Path, bbox_file: Path, training: bool):
        try:
            self.write_coco2yolo(annotation_file, bbox_file, training)
        except Exception as e:
            logging.error(f'{e.__class__.__name__} : {e.args}')
            if training:
                self.train_bbox_file.unlink(missing_ok=True)
            else:
                self.test_bbox_file.unlink(missing_ok=True)

    def write_coco2yolo(self, annotation_file: Path, bbox_file: Path, training: bool):
        images, classes = self.load_annotaion_file(annotation_file)
        classes = {
            class_name: index
            for index, class_name in enumerate(classes)
        }
        done = 0
        bbox_wh_img_size = []

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
                    bbox_wh_img_size.append(
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
        logging.info(f'{bbox_file.name} have {done} images')

        if training:
            self.calculate_anchor_box(bbox_wh_img_size)
            self.classes = list(classes.keys())

    def load_annotaion_file(self, file: Path):
        if file.suffix == '.pickle':
            with file.open('r') as f:
                logging.info(f'Start loading annotation file: {file.name}')
                data = pickle.load(f)
                np.random.shuffle(data['images'])
                np.random.shuffle(data['annotations'])
                logging.info(f'loading annotation file: {file.name} finish')
        elif file.suffix == '.json':
            with file.open('r') as f:
                logging.info(f'Start loading annotation file: {file.name}')
                data = json.load(f)
                np.random.shuffle(data['images'])
                np.random.shuffle(data['annotations'])
                logging.info(f'loading annotation file: {file.name} finish')
        else:
            raise RuntimeError('Unknown file suffix')

        return self._filter(data)

    def _filter(self, data):
        cats = {
            cat['id']: cat['name']
            for cat in data['categories'] if cat['name'] in self.classes
        }
        if len(cats) <= 0:
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

        self.anchors_tiny = self.cal_kmeans(w_h, 6)
        self.anchors = self.cal_kmeans(w_h, 9)
        self.anchors_v3 = self.anchors

    def cal_kmeans(self, w_h, k):
        x = np.array(w_h)
        kmeans = KMeans(n_clusters=k).fit(x)
        cluster = kmeans.cluster_centers_
        dot = [box[0] * box[1] for box in cluster]
        args = np.argsort(dot)
        arranged = np.zeros_like(cluster, int)

        for i, arg in enumerate(args):
            arranged[i] = cluster[arg]

        return arranged.reshape(-1).tolist()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO
    )

    sys_config_path = 'sys.conf'
    name = 'test3'
    cls = 'data/classes/person.txt'
    mk = MakeYoloConfig(name, cls, sys_config_path)
    mk.make()
    pprint(mk.__dict__)
