import json
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import argparse
from os import path

"""
    Convert coco dataset to yolo format
"""


def config_parser(cfg_path):
    """
    :param cfg_path: config file path
    :return: config
    """
    with open(cfg_path, 'r') as f:
        config = json.load(f)
    return config


def write_config(config: dict):
    """
    :param config: config dict
    :return: None
    """
    cfg_path = path.join('cfg', config['name']) + '.cfg'
    if path.exists(cfg_path):
        while True:
            ans = input('config file %s already exist , replace old config file? Y or N\n' % cfg_path)
            if ans == "Y" or ans == 'y':
                with open(cfg_path, 'w') as cfg:
                    cfg.write(json.dumps(config))
                    print('save new config file %s' % cfg_path)
                    break
            elif ans == "N" or ans == "n":
                print('config file %s did not save' % cfg_path)
                break
    else:
        with open(cfg_path, 'w') as cfg:
            cfg.write(json.dumps(config))
            print('write config file %s' % cfg_path)


def load_class(class_path):
    """
    :param class_path: classes file path
    :return: classes
    """
    if not path.exists(class_path):
        raise Exception('Classes file path does not effective')
    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def load_JSON_file(anno_file, classes):
    """
    :param anno_file: coco dataset annotation file path
    :param classes: classes name of dataset
    :return: dict:{
                Image_id:{
                    'file_name':name
                    'items':[  ->list
                        {
                            'bbox':bounding box
                            'category_name':class name of bbox
                        }
                    ]
                }
            }
    """

    with open(anno_file, 'r') as f:
        print('load %s...' % anno_file)
        data = json.load(f)

    cats = {
        cat['id']: cat['name']
        for cat in data['categories'] if cat['name'] in classes
    }
    annos = [anno for anno in data['annotations'] if anno['category_id'] in cats.keys()]

    imgs = {
        img['id']: {
            'file_name': img['file_name'],
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
    return imgs


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


def write_coco2yolo_file(coco_file, save_dir, classes, total, save_file_path, ):
    """
    :param coco_file: coco annotations file path
    :param save_dir: yolo format save path
    :param classes: classes name in coco dataset
    :param total: how many image need to write
    :param save_file_path: yolo format file save path
    :return: None
    """

    images = load_JSON_file(coco_file, classes)
    classes = {value: key for key, value in enumerate(classes)}
    data = open(save_file_path, 'w')
    done = 0
    w_h = []
    tbar = tqdm(total=total)
    if not path.isdir(save_dir):
        raise Exception(f'Save dir: %s does not effective dir.' % (save_dir))

    for img in images.values():
        file_path = path.join(save_dir, img['file_name'])
        if done >= total:
            break
        if not path.isfile(file_path):
            continue
        if len(img['items']) > 0:
            file_path += ' '
            data.write(file_path)
            for item in img['items']:
                xmin = int(item['bbox'][0])
                ymin = int(item['bbox'][1])
                xmax = xmin + int(item['bbox'][2])
                ymax = ymin + int(item['bbox'][3])
                w_h.append([xmax - xmin, ymax - ymin])
                xmin, ymin, xmax, ymax = str(xmin), str(ymin), str(xmax), str(ymax)
                label = str(classes[item['category_name']])
                data.write(','.join([xmin, ymin, xmax, ymax, label]) + ' ')
            data.write('\n')
            tbar.update(1)
            done += 1
    data.close()
    return w_h


def Main(args: argparse.ArgumentParser.parse_args):
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
    config['name'] = args.name
    config['model_path'] = path.join('checkpoints', args.name)
    config['weight_path'] = path.join('checkpoints', 'weights', args.name) + '.h5'
    # yolo
    config['frame_work'] = args.frame_work
    config['model_type'] = args.model_type
    config['size'] = args.size
    config['tiny'] = args.tiny
    config['YOLO']['CLASSES'] = load_class(args.class_path)
    config['YOLO']['INPUT_SIZE'] = args.size
    # train
    config['TRAIN']['INPUT_SIZE'] = args.size
    config['TRAIN']['ANNOT_PATH'] = path.join('data', 'annotations', args.name) + '.txt'
    # test
    config['TEST']['INPUT_SIZE'] = args.size
    config['TEST']['ANNOT_PATH'] = path.join('data', 'tests', args.name) + '.txt'

    w_h = write_coco2yolo_file(args.train_file, args.train_dir, config['YOLO']['CLASSES'], args.max_train,
                               config['TRAIN']['ANNOT_PATH'])
    write_coco2yolo_file(args.val_file, args.val_dir, config['YOLO']['CLASSES'], args.max_val,
                         config['TEST']['ANNOT_PATH'])
    anchor = get_anchor_box(w_h, args.tiny)
    anchor_tiny = get_anchor_box(w_h, args.tiny)
    config['YOLO']['ANCHORS'] = anchor
    config['YOLO']['ANCHORS_V3'] = anchor
    config['YOLO']['ANCHORS_TINY'] = anchor_tiny
    config['TRAIN']['PRETRAIN'] = args.pretrain

    write_config(config)
    """for k, v in config.items():
        print(k, ' : ', v)"""


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
