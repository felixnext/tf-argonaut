'''
Callback to validate multiple tasks during training.

author: Felix Geilert
'''

import os
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow import keras
from argonaut.components import heads

class MultiTaskValidation(Callback):
  '''Verifies the given network against various tasks.

  Args:
    base_model (tf.Model): The Base Model that underlies all different tasks
    tasks (List): List of task definitions that will be used to load models
    datasets (List): List of generated datasets from the Pipeline
    location (str): Path to the location of experiment data
    input_fn (function): Function to generate the input
  '''
  def __init__(self, base_model, tasks, datasets, location, input_fn):
    # set the file path
    self.location = location
    self.file_path = os.path.join(self.location, "eval_multitask.csv")
    self.task_path = os.path.join(self.location, "vals_{}.csv".format(tasks[-1]["id"]))

    # check for previous data to load
    self.df = pd.DataFrame()
    self.df.index = self.df.index.rename("epochs")
    if os.path.exists(self.file_path):
      self.df = pd.read_csv(self.file_path).set_index("epochs")
    self.df_task = pd.DataFrame()

    # store relevant data
    self.base_model = base_model
    self.tasks = tasks
    self.datasets = datasets
    self.input_fn = input_fn

  def on_batch_end(self, batch, logs={}):
    # create new row
    self.df_task = self.df_task.append(pd.Series(name="batch_{}".format(batch)))

    # add learning rate data (retrieve from optimizer)
    if "learning_rate" not in self.df_task:
      self.df_task["learning_rate"] = np.NaN
    self.df_task["learning_rate"][-1] = self.model.optimizer.lr.numpy()

    # add metrics to data (retrieve from last epoch)
    metrics = [] if "metrics" not in self.tasks[-1] else self.tasks[-1]["metrics"]
    metrics = ["loss"] + metrics
    for j, col in enumerate(metrics):
      if col not in self.df_task:
        self.df_task[col] = np.NaN
      self.df_task[col][-1] = logs.get(col)

  @tf.function
  def _eval_model(self, model, dataset, loss, metrics):
    '''Internal evaluation function to avoid compilation of the model.'''
    # iterate through all datapoints
    # TODO: avoid creation on non-first?
    avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
    met_vals = []
    for met in metrics:
      met_vals.append(tf.keras.metrics.Mean(name=met, dtype=tf.float32))

    # iterate through dataset
    for x, y in dataset:
      out = model(x, training=False)
      loss_val = loss(out, y)
      avg_loss.update_state(loss_val)
      # compute the metrics
      for i, met in enumerate(metrics):
        met_vals[i].update_state(tf.keras.metrics.get(met)(out, y))

    # generate the average output
    res = [avg_loss.result()]
    for i in range(len(met_vals)):
      res.append(met_vals[i].result())
    return res

  def on_epoch_end(self, epoch, logs=None):
    # FIXME: PERFORMANCE VALIDATE THIS FUNCTION!
    # add a new row
    self.df = self.df.append(pd.Series(name="epoch_{}".format(epoch)))
    # jump to new line to avoid clutter due to TF debug output (bug in TF 2.1)
    print("")

    # store the weights of the current model (to reload later)
    #base_path = os.path.join(self.location, "base_model_tmp.h5")
    #full_path = os.path.join(self.location, "model_tmp.h5")
    #self.base_model.save_weights(base_path, overwrite=True)
    #self.model.save_weights(full_path, overwrite=True)

    # iterate through relevant tasks
    for i, task in enumerate(self.tasks):
      # check if should be loaded (i.e. it is not the currently trained model)
      dataset = self.datasets[task["dataset"]]
      task_model = self.model
      metrics = [] if "metrics" not in task else task["metrics"]
      is_older = (i + 1 < len(self.tasks))
      # retrieve the loss function used for the model (used only for logging here)
      loss = getattr(tf.keras.losses, task["loss"])

      # only build the model for older than current task
      if is_older:
        # retrieve instance of head model (give name and use dataset to determine output size)
        head_fn = getattr(heads, task["head"])(dataset, name=task["id"])
        # generate the input to the model
        inputs = self.input_fn(dataset)
        rel_input = inputs
        # check if the input is a list (e.g. due to curriculum) - avoid that for testing
        if isinstance(inputs, list):
          rel_input = inputs[0]

        # execute the model
        main_latent = self.base_model(rel_input)
        head = head_fn(main_latent)
        task_model = keras.Model(inputs=inputs, outputs=head)

        # check if exists + check if weights are copied for the old data
        weight_path = os.path.join(self.location, "head_{}.h5".format(task["id"]))
        if not os.path.exists(weight_path):
          warnings.warn("Weight file for task ({}) is not found!".format(task["id"]))
        else:
          #self.base_model.load_weights(base_path)
          head_fn.load_weights(weight_path)

      # run evaluation
      if "type" in dataset and dataset.type == "tfdata":
        res = self._eval_model(task_model, dataset.test.batch(task["batch_size"]), loss, metrics)
        #res = task_model.evaluate(dataset.test.batch(task["batch_size"]), verbose=0)
      else:
        task_model.compile(loss=loss, metrics=metrics)
        res = task_model.evaluate(dataset.test.x, dataset.test.y, batch_size=task["batch_size"], verbose=0)

      # check if dataframe has columns for the task
      metrics = ["loss"] + metrics
      for j, col in enumerate(metrics):
        col_id = "{}_{}".format(task["id"], col)
        print("{:10}: {}".format(col_id, res[j]))
        if col_id not in self.df:
          self.df[col_id] = np.NaN
        self.df[col_id][-1] = res[j].numpy()
        # log data
        tf.summary.scalar(col_id, data=res[j], step=epoch)

      # add learning rate data
      if "learning_rate" not in self.df:
        self.df["learning_rate"] = np.NaN
      self.df["learning_rate"][-1] = self.model.optimizer.lr.numpy()

      # clear session (to avoid cluttering)
      # FIXME: remove from graph by avoid to remove actual model
      if is_older:
        del head_fn
        tf.keras.backend.clear_session()

    # CHECK: if required rebuild model
    #self.model.load_weights(full_path)

    # check epoch and save
    if epoch % 10 == 0:
      self.df.to_csv(self.file_path)
      self.df_task.to_csv(self.task_path)

  def on_train_end(self, logs):
    # write the results
    self.df.to_csv(self.file_path)
    self.df_task.to_csv(self.task_path)
