import json
import argparse
import numpy as np
import threading
import configparser
from pathlib import Path
import logging
from os import path
from sklearn.cluster import KMeans

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


def load_class(class_path):
    """
    :param class_path: classes file path
    :return: classes
    """
    class_path = Path(class_path)
    if not class_path.exists():
        raise Exception('Classes file path does not exists.')

    with class_path.open('r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def write_config(config: dict):
    """
    :param config: config dict
    :return: None
    """

    config_save_path = Path('configs') / (config['name'] + '.cfg')
    if config_save_path.exists():
        logging.warning(f'file : {config_save_path} is exist and will be replace.')
    with config_save_path.open('w') as f:
        json_object = json.dumps(config)
        f.write(json_object)


def load_JSON_file(anno_path: str, classes):
    """
    :param anno_path: coco dataset annotation file path
    :param classes: classes name of dataset
    :return: dict:{
                Image_id:{
                    'file_name':name
                    'width':image width
                    'height':image height
                    'items':[  ->list
                        {
                            'bbox':bounding box
                            'category_name':class name of bbox
                        }
                    ]
                },
                Classes:[classes name]
            }
    """

    with open(anno_path, 'r') as f:
        print('load %s...' % anno_path)
        data = json.load(f)
        np.random.shuffle(data['images'])

    cats = {
        cat['id']: cat['name']
        for cat in data['categories'] if cat['name'] in classes
    }
    annos = [anno for anno in data['annotations'] if anno['category_id'] in cats.keys()]

    imgs = {
        img['id']: {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height'],
            'items': []
        }
        for img in data['images']
    }

    for anno in annos:
        _id = anno['image_id']
        imgs[_id]['items'].append(
            {
                'bbox': anno['bbox'],
                'category_name': cats[anno['category_id']]
            }
        )

    cats = [name for name in cats.values()]

    return imgs, cats


def get_anchor_box(w_h, tiny):
    """
    :param w_h: [width, height] array
    :param tiny: if tiny = true , k =6 , else k =3
    :return: cluster centers (anchor box)
    """
    if tiny:
        k = 6
    else:
        k = 9

    x = np.array(w_h)
    kmeans = KMeans(n_clusters=k).fit(x)
    cluster = kmeans.cluster_centers_

    dot = [box[0] * box[1] for box in cluster]
    args = np.argsort(dot)
    arranged = np.zeros_like(cluster, dtype=np.int)

    for i, arg in enumerate(args):
        arranged[i] = cluster[arg]

    return arranged.reshape(-1).tolist()


def write_coco2yolo_file(anno_file_path, classes, data_set_dir, yolo_file_save_path, set_size, class_file_name='',
                         train=False):
    """
    :param anno_file_path: coco annotations file path
    :param data_set_dir: yolo format save path
    :param classes: [classes name]
    :param set_size: how many image need to write
    :param yolo_file_save_path: yolo format file save path
    :param class_file_name: class json file name
    :param train: if train will write class name to json file save in data/classes/class_file_name
    :return: None
    """

    anno_file_path = Path(anno_file_path)
    data_set_dir = Path(data_set_dir)
    yolo_file_save_path = Path(yolo_file_save_path)

    if not anno_file_path.is_file():
        logging.error(f'{anno_file_path} does not exists')
        raise FileNotFoundError
    if not data_set_dir.is_dir():
        logging.error(f'dir: {data_set_dir} not effective')
        raise NotADirectoryError(f'{data_set_dir}')
    if yolo_file_save_path.is_file():
        logging.warning(f'{yolo_file_save_path} will be replace')

    yolo_file = yolo_file_save_path.open('w')
    images, classes = load_JSON_file(anno_file_path, classes)
    classes = {
        name: index
        for index, name in enumerate(classes)
    }
    done = 0

    for img in images.values():
        file_path = Path(data_set_dir) / img['file_name']
        if done >= set_size:
            break
        if not file_path.exists():
            continue
        if len(img['items']) > 0:
            yolo_file.write(str(file_path))
            yolo_file.write(' ')
            for item in img['items']:
                xmin = int(item['bbox'][0])
                ymin = int(item['bbox'][1])
                xmax = xmin + int(item['bbox'][2])
                ymax = ymin + int(item['bbox'][3])
                xmin, ymin, xmax, ymax = str(xmin), str(ymin), str(xmax), str(ymax)
                label = str(classes[item['category_name']])
                yolo_file.write(','.join([xmin, ymin, xmax, ymax, label]) + ' ')
            yolo_file.write('\n')
            done += 1
    yolo_file.close()
    if train:
        class_file = Path('data') / 'classes' / (class_file_name + '.txt')
        with class_file.open('w') as f:
            f.write(json.dumps(classes))
    logging.info(f'file {yolo_file_save_path} has {done} images')


def Main(args):
    config = {
        # path
        'name': None,
        'model_path': None,
        'weight_path': None,

        # work environment
        'frame_work': None,
        'model_type': None,
        'size': None,
        'tiny': None,
        'max_output_size_per_class': 20,
        'max_total_size': 50,
        'iou_threshold': 0.25,
        'score_threshold': 0.5,

        # yolo option
        'YOLO': {
            'CLASSES': None,
            'ANCHORS': None,
            'ANCHORS_V3': None,
            'ANCHORS_TINY': None,
            'STRIDES': [8, 16, 32],
            'STRIDES_TINY': [16, 32],
            'XYSCALE': [1.2, 1.1, 1.05],
            'XYSCALE_TINY': [1.05, 1.05],
            'ANCHOR_PER_SCALE': 3,
            'IOU_LOSS_THRESH': 0.5,
        },

        # TRAIN
        'TRAIN': {
            'ANNOT_PATH': None,
            'BATCH_SIZE': 2,
            'INPUT_SIZE': None,
            'DATA_AUG': True,
            'LR_INIT': 1e-3,
            'LR_END': 1e-6,
            'WARMUP_EPOCHS': 2,
            'FISRT_STAGE_EPOCHS': 20,
            'SECOND_STAGE_EPOCHS': 30,
            'PRETRAIN': None,
        },

        # TEST
        'TEST': {
            'ANNOT_PATH': None,
            'BATCH_SIZE': 2,
            'INPUT_SIZE': None,
            'DATA_AUG': False,
            'SCORE_THRESHOLD': 0.25,
            'IOU_THRESHOLD': 0.5,
        },
    }

    # path
    paths = configparser.ConfigParser()
    paths.read('sys.ini')
    train_set_dir = paths['Paths']['train_set_dir']
    train_anno_path = paths['Paths']['train_annotation_path']
    val_set_dir = paths['Paths']['train_set_dir']
    val_anno_path = paths['Paths']['train_annotation_path']

    name = args.name
    model_path = Path('checkpoints') / name
    weight_path = Path('checkpoints') / 'weights' / (name + '.h5')
    classes = load_class(args.class_path)
    train_yolo_format_path = Path('data') / 'Sets' / 'train' / (name + '.txt')
    val_yolo_format_path = Path('data') / 'Sets' / 'test' / (name + '.txt')

    t1 = threading.Thread(
        target=write_coco2yolo_file,
        args=(
            str(train_anno_path),
            classes,
            train_set_dir,
            train_yolo_format_path,
            args.max_train,
            name,
            True
        )
    )

    t2 = threading.Thread(
        target=write_coco2yolo_file,
        args=(
            str(val_anno_path),
            classes,
            val_set_dir,
            val_yolo_format_path,
            args.max_test,
        )
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    config['name'] = args.name
    config['model_path'] = path.join('checkpoints', args.name)
    config['weight_path'] = path.join('checkpoints', 'weights', args.name) + '.h5'
    # yolo
    config['frame_work'] = args.frame_work
    config['model_type'] = args.model_type
    config['size'] = args.size
    config['tiny'] = args.tiny
    config['YOLO']['CLASSES'] = []
    config['YOLO']['INPUT_SIZE'] = args.size
    # train
    config['TRAIN']['INPUT_SIZE'] = args.size
    config['TRAIN']['ANNOT_PATH'] = path.join('data', 'annotations', args.name) + '.txt'
    # test
    config['TEST']['INPUT_SIZE'] = args.size
    config['TEST']['ANNOT_PATH'] = path.join('data', 'tests', args.name) + '.txt'

    # anchor = get_anchor_box(w_h, args.tiny)
    # anchor_tiny = get_anchor_box(w_h, args.tiny)
    # config['YOLO']['ANCHORS'] = anchor
    # config['YOLO']['ANCHORS_V3'] = anchor
    # config['YOLO']['ANCHORS_TINY'] = anchor_tiny
    # config['TRAIN']['PRETRAIN'] = args.pretrain

    write_config(config)


if __name__ == '__main__':
    paser = argparse.ArgumentParser(description='Make yolo config')
    paser.add_argument('-name', type=str, nargs='?', const=1, help='Model name')
    paser.add_argument('-size', type=int, nargs='?', const=1, default=416,
                       help='Detect image size in [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]')
    paser.add_argument('-model_type', type=str, nargs='?', const=1, default='yolov4',
                       help='Type of model \'yolov4\' or \'yolov3\'')
    paser.add_argument('-frame_work', type=str, nargs='?', const=1, default='tf', help='Work environment')
    paser.add_argument('-tiny', type=bool, nargs='?', const=1, default=False, help='Tiny model \'True\' or \'False\'')
    paser.add_argument('-class_path', type=str, nargs='?', const=1, default='', help='Classes file path')

    paser.add_argument('-train_dir', type=str, nargs='?', const=1, default=r'E:\TMPPPP\train2014\train2014',
                       help='Train data dir')
    paser.add_argument('-val_dir', type=str, nargs='?', const=1, default=r'E:\TMPPPP\val2014\val2014',
                       help='Val data dir')
    paser.add_argument('-train_file', type=str, nargs='?', const=1, default='instances_train2014.json',
                       help='COCO Train Annotation file path')
    paser.add_argument('-val_file', type=str, nargs='?', const=1, default='instances_val2014.json',
                       help='COCO Val Annotation file path')
    paser.add_argument('-max_train', type=int, nargs='?', const=1, default=1500, help='How many images want to train')
    paser.add_argument('-max_val', type=int, nargs='?', const=1, default=300, help='How many images want to test')
    paser.add_argument('-pretrain', type=str, nargs='?', const=1, default='', help='Pretrain weights path')
    args = paser.parse_args()
    Main(args)
