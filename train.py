import json
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, compute_loss, decode_train
from core.dataset import Dataset
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
from makeYoloConfig import JSON_parser
import argparse


def main(config_path):
    config = JSON_parser(config_path)

    trainset = Dataset(config, is_training=True)
    testset = Dataset(config, is_training=False)
    logdir = os.path.join('data', 'log', config['name'])
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = config['TRAIN']['FISRT_STAGE_EPOCHS']
    second_stage_epochs = config['TRAIN']['SECOND_STAGE_EPOCHS']
    IOU_LOSS_THRESH = config['YOLO']['IOU_LOSS_THRESH']
    LR_INIT, LR_END = config['TRAIN']['LR_INIT'], config['TRAIN']['LR_END']
    INPUT_SIZE = config['TRAIN']['INPUT_SIZE']
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = config['TRAIN']['WARMUP_EPOCHS'] * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    input_layer = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(config)
    freeze_layers = utils.load_freeze_layer(config['model_type'], config['tiny'])
    feature_maps = YOLO(input_layer, NUM_CLASS, config['model_type'], config['tiny'])

    if config['tiny']:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i,
                                           XYSCALE)
            else:
                bbox_tensor = decode_train(fm, INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i,
                                           XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i,
                                           XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i,
                                           XYSCALE)
            else:
                bbox_tensor = decode_train(fm, INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i,
                                           XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()

    if config['TRAIN']['PRETRAIN']:
        if config['TRAIN']['PRETRAIN'].endswith('weights'):
            utils.load_weights(model, config['TRAIN']['PRETRAIN'], config['model_type'], config['tiny'])
        else:
            model.load_weights(config['TRAIN']['PRETRAIN'])
        print('Train from %s' % (config['TRAIN']['PRETRAIN']))
    else:
        print("Training from scratch")



    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
        os.makedirs(logdir)
    else:
        os.makedirs(logdir)
    writer = tf.summary.create_file_writer(logdir)

    # define training step function
    # @tf.function
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS,
                                          IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * LR_INIT
            else:
                lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
                config['TRAIN']['LR_INIT'] = float(lr)
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()

    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS,
                                          IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))

    for epoch in range(first_stage_epochs + second_stage_epochs):
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
            train_step(image_data, target)
        for image_data, target in testset:
            test_step(image_data, target)
        model.save_weights(config['weight_path'])
        config['TRAIN']['PRETRAIN'] = config['weight_path']
        with open(config_path, 'w') as f:
            json.dump(config, fp=f)

        print('model save %s' % config['weight_path'])


if __name__ == '__main__':
    paser = argparse.ArgumentParser(description='Train model using config file')
    paser.add_argument('--config', type=str, help='config file path')
    args = paser.parse_args()
    main(args.config)
