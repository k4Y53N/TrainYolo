from .yolov4 import YOLO, decode_train, decode, filter_boxes
from .configer import YOLOConfiger
import tensorflow as tf


def build_train_model(configer: YOLOConfiger):
    tiny = configer.tiny
    size = configer.size
    input_layer = tf.keras.layers.Input([size, size, 3])
    strides = configer.strides
    anchors = configer.anchors
    num_class = configer.num_class
    xyscale = configer.xyscale
    feature_maps = YOLO(input_layer, num_class, configer.model_type, configer.tiny)
    bbox_tensors = []
    if tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, size // 16, num_class, strides, anchors, i,
                                           xyscale)
            else:
                bbox_tensor = decode_train(fm, size // 32, num_class, strides, anchors, i,
                                           xyscale)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, size // 8, num_class, strides, anchors, i,
                                           xyscale)
            elif i == 1:
                bbox_tensor = decode_train(fm, size // 16, num_class, strides, anchors, i,
                                           xyscale)
            else:
                bbox_tensor = decode_train(fm, size // 32, num_class, strides, anchors, i,
                                           xyscale)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    return tf.keras.Model(input_layer, bbox_tensors)


def build_model(configer: YOLOConfiger, training=False):
    if training:
        return build_train_model(configer)

    size = configer.size
    frame_work = configer.frame_work
    tiny = configer.tiny
    model_type = configer.model_type
    score_threshold = configer.score_threshold
    num_class = configer.num_class
    weight_path = configer.weight_path
    strides = configer.strides
    anchors = configer.anchors
    xyscale = configer.xyscale
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
    model.load_weights(weight_path)

    return model
