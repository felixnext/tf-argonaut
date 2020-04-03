'''
Callback function that allows to generate curriculum on the dataset.
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from bunch import Bunch

class CurriculumCallback(Callback):
  '''Defines a callback that adjusts the curriculum learning function

  Args:
    variable (tf.Variable): Variable that adjusts the curriculum usage
    max_epochs (int): The number of the last epoch in the training process (Note: this may be affected by continuation of training)
    max_batches (int): In case of batch mode the number of batches per epoch (if None, will be determined in the first epoch)
    function (str): Name of the conversion function to use (options are: [linear, sigmoid, exp])
    start_epoch (int): Number of epoch to start the conversion process
    mode (str): Mode of the variable update (either `epoch` or `batch`)
    params (dict): additional parameters based on the function (`slope` for sigmoid, `exp` for exp)
  '''
  def __init__(self, variable, max_epochs, max_batches=None, function="linear", start_epoch=0, mode="epoch", **params):
    super().__init__()
    # store relevant data
    self.var = variable
    self.mode = mode
    self.start_epoch = start_epoch
    self.max_epochs = max_epochs - start_epoch
    self.max_batches = max_batches
    if function not in ['linear', 'sigmoid', 'exp']:
      raise ValueError("Unkown computation function ({}) selected for curriculum".format(function))
    self.cur_fn = function
    self.fn_params = params
    self._mb = 0
    self.init_val = variable.numpy()

  @classmethod
  def filter(cls, x, y, var):
    '''Takes relevant input and checks if it is relevant to curriculum.'''
    # check if curriculum value is provided
    if len(x) < 2:
      return True
    # apply the relevant function
    return tf.less(x[1], var)

  @classmethod
  def create(cls, dataset, init_val=0.5, **args):
    '''Creates the required prequesites in the dataset.

    Args:
      dataset (tf.Data): Dataset that should be modified
      args (dict): List of arguments (same as in class Constructor)

    Returns:
      callback: The created callback
      dataset: The updated dataset (filtered for current values)
    '''
    # defines the variable and creates curriculum
    var = tf.Variable(init_val, dtype=tf.float32, trainable=False)
    callback = CurriculumCallback(var, **args)

    # filter dataset based on current variable state
    # CHECK: Check if updates in filter still works if the dataset is buffered (through shuffling)
    ds = Bunch(dataset.copy())
    ds.update( train = dataset.train.filter(lambda x, y: cls.filter(x, y, var)) )

    # return relevant data
    return callback, ds

  def _calc_val(self, epoch, batch):
    '''Calculates the next position for the complexity value and applies it.'''
    # calculate the total progress position
    epoch_prog = (epoch - self.start_epoch) / self.max_epochs
    batch_prog = 0 if batch is None else (batch / self.max_batches)
    prog = epoch_prog + (batch_prog / self.max_epochs)

    # select the function and calculate position
    nvar = 0
    if self.cur_fn == "linear":
      nvar = prog
    elif self.cur_fn == "sigmoid":
      k = 20 if "slope" not in self.fn_params else self.fn_param["slope"]
      nvar = 1 / (1 + np.exp(-k * (nvar - 0.5)))
    elif self.cur_fn == "exp":
      exp = 1.5 if "exp" not in self.fn_params else self.fn_params["exp"]
      nvar = np.exp(nvar, exp)

    # update variable
    nvar = min(max(nvar, 0.), 1.)
    nvar = self.init_val + ((1 - self.init_val) * nvar)
    self.var.assign(nvar)
    # log data
    steps = epoch
    if batch is not None:
        steps = self.max_batches * (epoch - 1) + batch
    tf.summary.scalar("curriculum", self.var, step=steps)

  def on_epoch_start(self, epoch, logs=None):
    # update some starting vars
    self._epoch = epoch
    if self.max_batches is None:
      self._mb = 0

  def on_batch_end(self, batch, logs=None):
    # call the update
    if self.mode == "batch" and self.max_batchs is not None:
      self._calc_val(self._epoch, batch)
    # count potential batches
    if self.max_batches is None:
      self._mb += 1

  def on_epoch_end(self, epoch, logs=None):
    # call the update
    if self.mode == "epoch":
      self._calc_val(epoch, None)
    # set the potential max batches
    if self.max_batches is None:
      self.max_batches = self._mb
