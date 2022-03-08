import tensorflow as tf
from core.yolov4 import YOLO, decode, filter_boxes
from core.utils import load_config
import argparse
import os
import json
import shutil


def build_model(config):
    strides, anchors, num_class, xyscale = load_config(config)
    size = config['size']
    frame_work = config['frame_work']
    model_type = config['model_type']
    tiny = config['tiny']
    score_threshold = config['score_threshold']
    input_layer = tf.keras.layers.Input([size, size, 3])
    feature_maps = YOLO(input_layer, num_class, model_type, tiny)
    bbox_tensors = []
    prob_tensors = []
    if tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, size // 16, num_class, strides, anchors, i, xyscale, frame_work)
            else:
                output_tensors = decode(fm, size // 32, num_class, strides, anchors, i, xyscale, frame_work)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, size // 8, num_class, strides, anchors, i, xyscale, frame_work)
            elif i == 1:
                output_tensors = decode(fm, size // 16, num_class, strides, anchors, i, xyscale, frame_work)
            else:
                output_tensors = decode(fm, size // 32, num_class, strides, anchors, i, xyscale, frame_work)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    if frame_work == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=score_threshold,
                                        input_shape=tf.constant([size, size]))
        pred = tf.concat([boxes, pred_conf], axis=-1)

    model = tf.keras.Model(input_layer, pred)
    return model


def save_tf(config):
    size = config['size']
    frame_work = config['frame_work']
    strides, anchors, num_class, xyscale = load_config(config)
    model_type = config['model_type']
    tiny = config['tiny']
    score_threshold = config['score_threshold']
    weight_path = config['weight_path']
    model_path = config['model_path']

    input_layer = tf.keras.layers.Input([size, size, 3])
    feature_maps = YOLO(input_layer, num_class, model_type, tiny)
    bbox_tensors = []
    prob_tensors = []
    if tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, size // 16, num_class, strides, anchors, i, xyscale,
                                        frame_work)
            else:
                output_tensors = decode(fm, size // 32, num_class, strides, anchors, i, xyscale,
                                        frame_work)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, size // 8, num_class, strides, anchors, i, xyscale,
                                        frame_work)
            elif i == 1:
                output_tensors = decode(fm, size // 16, num_class, strides, anchors, i, xyscale,
                                        frame_work)
            else:
                output_tensors = decode(fm, size // 32, num_class, strides, anchors, i, xyscale,
                                        frame_work)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    if frame_work == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=score_threshold,
                                        input_shape=tf.constant([size, size]))
        pred = tf.concat([boxes, pred_conf], axis=-1)

    model = tf.keras.Model(input_layer, pred)
    model.load_weights(weight_path)
    model.save(model_path)


def main(config):
    model_path = config.get('model_path', '')
    weight_path = config.get('weight_path', '')

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(weight_path)

    if os.path.isdir(config['model_path']):
        print(f'WARN: {model_path} exists and will be remove')
        shutil.rmtree(model_path)
        os.makedirs(model_path)

    model = build_model(config)
    model.load_weights(weight_path)
    model.save(model_path)


if __name__ == '__main__':
    paser = argparse.ArgumentParser(description='Save Tensorflow Yolo model')
    paser.add_argument('--config', required=True, type=str, help='Config file path')
    args = paser.parse_args()

    try:
        config_path = args.config
        if not os.path.isfile(config_path):
            raise FileNotFoundError(config_path)

        with open(config_path, 'r') as f:
            config = json.load(f)
        main(config)
    except Exception as e:
        print(e)
