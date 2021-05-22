from tensorflow import keras
from core.yolov4 import compute_loss
import tensorflow as tf


class yolo_loss(keras.losses.Loss):
    def __init__(self, freeze_layers, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, **kwargs):
        self.freeze_layers = freeze_layers
        self.STRIDES = STRIDES
        self.NUM_CLASS = NUM_CLASS
        self.IOU_LOSS_THRESH = IOU_LOSS_THRESH
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        giou_loss = conf_loss = prob_loss = 0

        for i in range(len(self.freeze_layers)):
            conv, pred = y_pred[i * 2], y_pred[i * 2 + 1]
            loss_items = compute_loss(pred, conv, y_true[i][0], y_true[i][1], STRIDES=self.STRIDES,
                                      NUM_CLASS=self.NUM_CLASS,
                                      IOU_LOSS_THRESH=self.IOU_LOSS_THRESH, i=i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = tf.reduce_sum([giou_loss, conf_loss, prob_loss])

        return total_loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}
