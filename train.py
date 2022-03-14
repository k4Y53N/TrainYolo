import tensorflow as tf
import numpy as np
import argparse
from core.yolov4 import YOLO, compute_loss, decode_train
from core.utils import freeze_all, unfreeze_all
from core.dataset import Dataset
from core.configer import YOLOConfiger
from core import utils


def main(config_path):
    configer = YOLOConfiger(config_path)
    model_type = configer.model_type
    tiny = configer.tiny
    trainset = Dataset(configer, is_training=True)
    testset = Dataset(configer, is_training=False)
    logdir = configer.logdir
    isfreeze = False
    steps_per_epoch = len(trainset)
    init_epoch = configer.init_epoch
    first_stage_epochs = configer.first_stage_epochs
    second_stage_epochs = configer.second_stage_epochs
    total_stage_epochs = first_stage_epochs + second_stage_epochs
    iou_loss_thresh = configer.iou_loss_thresh
    lr_init = configer.lr_init
    lr_end = configer.lr_end
    size = configer.size
    warmup_steps = configer.warmup_epochs * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    global_steps = tf.Variable(init_epoch * steps_per_epoch + 1, trainable=False, dtype=tf.int64)
    input_layer = tf.keras.layers.Input([size, size, 3])
    strides = configer.strides
    anchors = configer.anchors
    num_class = configer.num_class
    xyscale = configer.xyscale
    freeze_layers = configer.freeze_layers
    feature_maps = YOLO(input_layer, num_class, configer.model_type, configer.tiny)

    pretrain_path = configer.pre_train_file_path
    weight_path = configer.weight_path
    if tiny:
        bbox_tensors = []
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
        bbox_tensors = []
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

    model = tf.keras.Model(input_layer, bbox_tensors)

    if pretrain_path:
        if pretrain_path.endswith('weights'):
            utils.load_weights(model, pretrain_path, model_type, tiny)
        else:
            model.load_weights(pretrain_path)
        print('Train from %s' % (pretrain_path))
    else:
        print("Training from scratch")

    optimizer = tf.keras.optimizers.Adam()
    writer = tf.summary.create_file_writer(logdir)

    def train_step(image_data, target, epoch=0):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=strides, NUM_CLASS=num_class,
                                          IOU_LOSS_THRESH=iou_loss_thresh, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print("\r=> STEP %4d/%4d   epoch: %d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                  "prob_loss: %4.2f   total_loss: %5.2f" % (global_steps, total_steps, epoch, optimizer.lr.numpy(),
                                                            giou_loss, conf_loss,
                                                            prob_loss, total_loss), end='', flush=True)
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * lr_init
            else:
                lr = lr_end + 0.5 * (lr_init - lr_end) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
                configer.update_lr_init(float(lr))
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()

    def test_step(image_data, target, epoch=0):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=False)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=strides, NUM_CLASS=num_class,
                                          IOU_LOSS_THRESH=iou_loss_thresh, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            print("\r=> TEST STEP %4d  epoch: %d  giou_loss: %4.2f   conf_loss: %4.2f   "
                  "prob_loss: %4.2f   total_loss: %5.2f" % (global_steps, epoch, giou_loss, conf_loss,
                                                            prob_loss, total_loss), end='', flush=True)

    for epoch in range(init_epoch, total_stage_epochs):
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        for image_data, target in trainset:
            train_step(image_data, target, epoch + 1)
        print()
        for image_data, target in testset:
            test_step(image_data, target, epoch + 1)
        print()
        model.save_weights(weight_path)
        configer.update_pre_train_file_path(weight_path)
        configer.update_init_epoch(epoch + 1)
        configer.save()
        print('model save %s' % weight_path)


if __name__ == '__main__':
    paser = argparse.ArgumentParser(description='Train model using config file')
    paser.add_argument('--config', type=str, help='config file path')
    args = paser.parse_args()
    main(args.config)
