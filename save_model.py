import tensorflow as tf
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils


def save_tf(config):
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(config)

    input_layer = tf.keras.layers.Input([config['size'], config['size'], 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, config['model_type'], config['tiny'])
    bbox_tensors = []
    prob_tensors = []
    if config['tiny']:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, config['size'] // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        config['frame_work'])
            else:
                output_tensors = decode(fm, config['size'] // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        config['frame_work'])
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, config['size'] // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        config['frame_work'])
            elif i == 1:
                output_tensors = decode(fm, config['size'] // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        config['frame_work'])
            else:
                output_tensors = decode(fm, config['size'] // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        config['frame_work'])
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    if config['frame_work'] == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=config['score_threshold'],
                                        input_shape=tf.constant([config['size'], config['size']]))
        pred = tf.concat([boxes, pred_conf], axis=-1)

    model = tf.keras.Model(input_layer, pred)
    model.load_weights(config['weight_path'])
    #utils.load_weights(model, config['weight_path'], model_name=config['model_type'], is_tiny=config['tiny'])
    tf.saved_model.save(model, config['model_path'])


def model_test(config):
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(config)

    input_layer = tf.keras.layers.Input([config['size'], config['size'], 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, config['model_type'], config['tiny'])
    bbox_tensors = []
    prob_tensors = []
    if config['tiny']:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, config['size'] // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        config['frame_work'])
            else:
                output_tensors = decode(fm, config['size'] // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        config['frame_work'])
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, config['size'] // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        config['frame_work'])
            elif i == 1:
                output_tensors = decode(fm, config['size'] // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        config['frame_work'])
            else:
                output_tensors = decode(fm, config['size'] // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        config['frame_work'])
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    if config['frame_work'] == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=config['score_threshold'],
                                        input_shape=tf.constant([config['size'], config['size']]))
        pred = tf.concat([boxes, pred_conf], axis=-1)

    model = tf.keras.Model(input_layer, pred)
    return model


if __name__ == '__main__':
    from makeConfig import JSON_parser
    cfg = JSON_parser('cfg\\person.cfg')
    save_tf(cfg)