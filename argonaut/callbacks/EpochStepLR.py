'''
Implementation of stepwise learningrate decay after epochs
'''

import warnings
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np

class EpochStepLR(Callback):
  """Reduceds the learning rate in steps after certain epochs

  Args:
    boundaries (list): List of boudary values (either percentage or epoch number) after which to adjust
    values (list): List of learning values adapted after a boudary is reached (len should be boundaries + 1)
    start_epoch (int): For correct calculation after warm start
    max_epoch (int): Number of the maximum epoch to train (in case of percentage)
    is_percentage (bool): Defines if the boundaries are given as a percentage
  """

  def __init__(self, boundaries, values, start_epoch=0, max_epoch=None, is_percentage=False):
    super(EpochStepLR, self).__init__()

    self.is_perc = is_percentage
    self.lr_vals = values
    self.lr_bounds = boundaries
    self.pos = 0
    self.max_epoch = max_epoch
    self.start_epoch = start_epoch
    # safty checks
    if self.is_perc == True and max_epoch is None:
      warnings.warn("Percentage is set, but no max_epoch is given, assume 100")
      self.max_epoch = 100
    if len(self.lr_vals) != len(self.lr_bounds) + 1:
      raise ValueError("Values should have exactly one value more than boundaries!")

  def _set_lr(self, epoch):
    # save value
    old_pos = self.pos
    cur_ep = epoch - self.start_epoch

    # check if end reached
    if self.pos < len(self.lr_bounds):
      if self.is_perc == True and (cur_ep / self.max_epoch) >= self.lr_bounds[self.pos]:
        self.pos += 1
      elif self.is_perc == False and cur_ep >= self.lr_bounds[self.pos]:
        self.pos += 1

    tf.summary.scalar("epoch learning rate", data=self.lr_vals[self.pos], step=epoch)

    # change value
    if old_pos < self.pos:
      K.set_value(self.model.optimizer.lr, self.lr_vals[self.pos])

  def on_train_begin(self, logs={}):
    logs = logs or {}
    self._set_lr(0)

  def on_epoch_begin(self, epoch, logs=None):
    logs = logs or {}
    self._set_lr(epoch)
