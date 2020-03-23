'''
Implementation of stepwise learningrate decay after epochs
'''

import warnings
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np

class PlateauLR(Callback):
  """Reduceds the learning rate if certain metric plateaus for a given time

  Args:
    monitor (str): Metric to check
    min_delta (float): Maximum allowed absolute change from previous epoch
    patience (int): Number of epochs to watch
    decay (float): Factor by which the learning rate is decayed. If list use current decay position
  """

  def __init__(self, monitor='val_loss', min_delta=0, patience=0, decay=3):
    super(PlateauLR, self).__init__()

    # define values
    self.cur_lr = None
    self.min_delta = min_delta
    self.patience = patience
    self.decay = decay
    self.monitor = monitor
    self.last_val = None
    self.streak = 0
    self.list_pos = 0

  def _set_lr(self, epoch, logs):
    # summary data
    tf.summary.scalar("plateau_learning_rate", data=self.cur_lr, step=epoch)

    # save value
    met = logs[self.monitor]

    # check for changes in value
    if self.last_val is not None:
      if self.last_val - met <= self.min_delta:
        self.streak += 1
      else:
        self.last_val = met
        self.streak = 0

      # update the current learning rate
      if self.streak > self.patience:
        if isinstance(self.decay, list):
          self.cur_lr = self.cur_lr / self.decay[self.list_pos]
          self.list_pos = min(len(self.decay) - 1, self.list_pos + 1)
        else:
          self.cur_lr = self.cur_lr / self.decay
        self.streak = 0
    else:
      self.last_val = met

    # change value
    K.set_value(self.model.optimizer.lr, self.cur_lr)

  def on_train_begin(self, logs={}):
    logs = logs or {}
    self.cur_lr = self.model.optimizer.lr.numpy()

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    self._set_lr(epoch, logs)
