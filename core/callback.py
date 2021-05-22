from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import numpy as np
from core.utils import freeze_all, unfreeze_all


class WarmupLearningRate(keras.callbacks.Callback):
    def __init__(self, warmup_steps, total_steps, LR_INIT, LR_END):
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.LR_INIT = LR_INIT
        self.LR_END = LR_END

    def on_train_batch_end(self, batch, logs=None):
        self.global_step += 1
        if self.global_step < self.warmup_steps:
            lr = self.global_steps / self.warmup_steps * self.LR_INIT
        else:
            lr = self.LR_END + 0.5 * (self.LR_INIT - self.LR_END) * (
                (1 + tf.cos((self.global_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi))
            )
        K.set_value(self.model.optimizer.lr, lr)


class FreezeLayer(keras.callbacks.Callback):
    def __init__(self, first_stage_epochs, freeze_layers, isfreeze=False):
        self.first_stage_epochs = first_stage_epochs
        self.freeze_layers = freeze_layers
        self.isfreeze = isfreeze

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.first_stage_epochs:
            if not self.isfreeze:
                self.isfreeze = True
                for name in self.freeze_layers:
                    freeze = self.model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= self.first_stage_epochs:
            if self.isfreeze:
                self.isfreeze = False
                for name in self.freeze_layers:
                    freeze = self.model.get_layer(name)
                    unfreeze_all(freeze)
