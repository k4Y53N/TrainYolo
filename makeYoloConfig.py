import json
import argparse
import numpy as np
import threading
import configparser
import logging
from sklearn.cluster import KMeans
from pathlib import Path
from scripts.utils import printdic

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


def write_config(config: dict, dst):
    """
    :param config: config dict
    :param dst: destination
    :return: None
    """

    config_save_path = Path(dst)
    if config_save_path.exists():
        logging.warning(f'file : {config_save_path} is exist and will be replace.')
    with config_save_path.open('w') as f:
        json_object = json.dumps(config)
        f.write(json_object)
    logging.info(f'Config save at {config_save_path}')


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
        np.random.shuffle(data['annotations'])

    cats = {
        cat['id']: cat['name']
        for cat in data['categories'] if cat['name'] in classes
    }
    if len(cats) <= 0:
        raise Exception('Cant fine any classes in annotation file')

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
    k = 6 if tiny else 9

    x = np.array(w_h)
    kmeans = KMeans(n_clusters=k).fit(x)
    cluster = kmeans.cluster_centers_

    dot = [box[0] * box[1] for box in cluster]
    args = np.argsort(dot)
    arranged = np.zeros_like(cluster, dtype=np.int)

    for i, arg in enumerate(args):
        arranged[i] = cluster[arg]

    return arranged.reshape(-1).tolist()


def cal_anchors(set_file_path):
    """
    calculate anchor box
    @param set_file_path: dataset annotation file path
    @return: [[anchor box with k=9], [anchor box with k=6]]
    """
    set_file_path = Path(set_file_path)

    if not set_file_path.is_file():
        raise FileNotFoundError(f'file {set_file_path} is not exists')
    f = set_file_path.open('r')

    w_h = [
        [int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])]
        for line in f.readlines()
        for obj in line.strip().split(' ', )[1:]
        for bbox in [obj.split(',')]
    ]
    f.close()
    return [
        get_anchor_box(w_h, tiny=False),
        get_anchor_box(w_h, tiny=True)
    ]


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
        if class_file.exists():
            logging.warning(f'file {class_file} will be replace')
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
    name = args.name
    # path
    paths = configparser.ConfigParser()
    paths.read('sys.ini')
    train_set_dir = paths['Paths']['train_set_dir']
    train_anno_path = paths['Paths']['train_annotation_path']
    val_set_dir = paths['Paths']['val_set_dir']
    val_anno_path = paths['Paths']['val_annotation_path']

    model_path = Path(paths['Save_dir']['checkpoints']) / name
    weight_path = Path(paths['Save_dir']['weights']) / (name + '.h5')
    train_yolo_format_save_path = Path(paths['Save_dir']['train_processed_data']) / (name + '.txt')
    val_yolo_format_save_path = Path(paths['Save_dir']['test_processed_data']) / (name + '.txt')
    config_save_path = Path(paths['Save_dir']['yolo_config_path']) / (name + '.cfg')

    train_epoch_size = args.train_size
    val_epoch_size = args.val_size
    classes = load_class(args.classes)

    t1 = threading.Thread(
        target=write_coco2yolo_file,
        args=(
            str(train_anno_path),
            classes,
            train_set_dir,
            train_yolo_format_save_path,
            train_epoch_size,
            name,
            True
        ), daemon=True
    )

    t2 = threading.Thread(
        target=write_coco2yolo_file,
        args=(
            str(val_anno_path),
            classes,
            val_set_dir,
            val_yolo_format_save_path,
            val_epoch_size,
        ), daemon=True
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    classes_file_path = Path('data') / 'classes' / (name + '.txt')
    if not classes_file_path.is_file():
        raise Exception('Cant handle this data set please check out the classes name file')
    with classes_file_path.open() as f:
        classes = json.load(f)
    anchor = cal_anchors(str(train_yolo_format_save_path))
    classes = [_class for _class in classes.keys()]

    config['name'] = name
    config['model_path'] = str(model_path)
    config['weight_path'] = str(weight_path)
    config['frame_work'] = args.frame_work
    config['size'] = args.size
    config['model_type'] = args.model
    config['tiny'] = args.tiny
    config['YOLO']['CLASSES'] = classes
    config['YOLO']['ANCHORS'] = anchor[0]
    config['YOLO']['ANCHORS_V3'] = anchor[0]
    config['YOLO']['ANCHORS_TINY'] = anchor[1]
    config['TRAIN']['ANNOT_PATH'] = str(train_anno_path)
    config['TRAIN']['INPUT_SIZE'] = args.size
    config['TEST']['ANNOT_PATH'] = str(val_anno_path)
    config['TEST']['INPUT_SIZE'] = args.size

    print('-------------------CONFIG-------------------')
    printdic(config)
    print('--------------------END---------------------')
    write_config(config, dst=str(config_save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make yolo config and preprocess dataset')
    parser.add_argument('-n', '--name', required=False, type=str, help='Model name')
    parser.add_argument('-c', '--classes', required=True, type=str, help='classes name file path')
    parser.add_argument('-s', '--size', type=int, default=416,
                        choices=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
                        help='Image input size')
    parser.add_argument('-m', '--model', type=str, default='yolov4', choices=['yolov4', 'yolov3'], help='Model type')
    parser.add_argument('-f', '--frame_work', type=str, default='tf', choices=['tf', 'trt', 'tflite'],
                        help='Frame work')
    parser.add_argument('-t', '--tiny', type=bool, default=False, help='Tiny model?')
    parser.add_argument('-ts', '--train_size', type=int, default=1500, help='Train epoch size')
    parser.add_argument('-vs', '--val_size', type=int, default=300, help='Val epoch size')
    args = parser.parse_args()
    Main(args)
