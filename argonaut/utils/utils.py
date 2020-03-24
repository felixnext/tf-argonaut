'''
Defines generally used items in the dataset.

author: Felix Geilert
'''

import os
import shutil
import re
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

class Entry():
  def __init__(self, name, desc, params):
    self.name = name
    self.description = desc
    self.params = params
    self.time = time()

  def __str__(self):
    res = []
    # handle main data
    res.append('\033[1m{} - {}\033[0m'.format(datetime.fromtimestamp(self.time), self.name))
    if self.description is not None:
      res.append('> {}'.format(self.description.replace("\n", "\n> ")))

    # handle params
    if self.params is not None:
      res.append('> PARAMS:')
      # check types of params
      if isinstance(self.params, dict):
        for key in self.params.keys():
          res.append('> - {:5}: {}'.format(key, self.params[key]))
      elif isinstance(self.params, list) or isinstance(self.params, np.ndarray):
        for x in self.params:
          res.append('> - {}'.format(x))
      elif isinstance(self.params, pd.DataFrame):
        # NOTE: adjust
        res.append('> {}'.format(self.params))
      elif isinstance(self.params, str) or isinstance(self.params, int):
        res.append('> > {}'.format(self.params))
      elif isinstance(self.params, float):
        res.append('> > {:.4f}'.format(self.params))
      else:
        res.append('> > \033[93munkown datatype\033[0m - {}'.format(self.params))

    return "\n".join(res)

class Summary():
  '''Defines a summary class that stored relevant information to all changes done in the experiment.'''

  def __init__(self, name=None):
    self.entries = []
    # define the name
    if name is None:
      self.name = "Model_{}".format(int(time()))
    else:
      self.name = name

  def add(self, name, description, params=None, default_name=None):
    '''Adds a new entry to the summary.

    Args:
      title (str): Name of the entry
      description (str): description of the entry (type of operations performed)
      params (list): Object (either list, bunch or dict) with relevant parameters for the operation
    '''
    # simply add
    self.entries.append(Entry(name if name is not None else default_name, description, params))

  def add_str(self, str):
    '''Adds a single string message to the summary.'''
    self.add("message", str, None)

  def __str__(self):
    '''Shows the entire summary log.'''
    res = []
    res.append("\n--- Summary ---")

    for entry in self.entries:
      res.append(str(entry))

    res.append("---------------\n")

    return "\n".join(res)

  def generate_summary(self, folder, model=None, histories=None, horizontal=False, overwrite=True):
    '''Generates a summary and stores it at the given location.

    Args:
      folder (str): Folder where to store the summary
      model (tf.keras.Model): Model that is trained
      histories (tf.keras.History): dict of histories that contain information about individual training processes
      validation (tf.Results): Results from the model evaluation for every task at the end of the training process
      horizontal (bool): Defines if the model should be plotted horizontal
      overwrite (bool): Defines if old data should be cleared
    '''
    # generate folder structure
    #target = os.path.join(folder, self.name)
    target = os.path.join(folder, "results")
    if os.path.exists(target) and overwrite is True:
      shutil.rmtree(target)
    if not os.path.exists(target):
      os.makedirs(target)

    if model is not None:
      # store the image
      # FIXME: Appears not work (due to get_config of the actual models) - For now display in tensorboard...
      #plot_model(model, os.path.join(target, 'model.png'), show_shapes=True, expand_nested=True, dpi=128, rankdir='LR' if horizontal else 'TB')

      # generate summary
      with open(os.path.join(target, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda s: f.write("{}\n".format(s)))

    # generate log
    with open(os.path.join(target, 'experiment.log'), 'w') as f:
      f.write(re.sub("\\033\\[[0-9]+m", "", str(self)))

    hist_data = []
    # generate training data and plots
    if histories is not None:
      for task_id in histories:
        # Plot training & validation accuracy values
        history = histories[task_id]
        hist = history.history
        used = {}
        # iterate through all keys in histogram
        for key in hist:
          # check if used
          if key in used:
            continue
          # plot data
          plt.plot(hist[key])
          hist_data.append(np.array([task_id, key] + list(hist[key])))
          used[key] = True
          has_val = False
          # check for validation data
          vkey = "val_{}".format(key)
          if vkey in hist:
            plt.plot(hist[vkey])
            hist_data.append(np.array([task_id, vkey] + list(hist[vkey])))
            has_val = True
            used[vkey] = True

          # configure plot
          plt.title('Model {}'.format(key))
          plt.ylabel(key)
          plt.xlabel('Epoch')
          if has_val is True:
            plt.legend(['Train', 'Test'], loc='upper left')
          plt.savefig(os.path.join(target, '{}_{}.png'.format(task_id, key)))
          # clear for next picture
          plt.clf()

        # generate pandas elements and store csv
        df = pd.DataFrame(np.array(hist_data))
        df.to_csv(os.path.join(target, "train_histories.csv"))

      # FEAT: generate combined graphs and csv (to integrate in HTML file)

    # generate index-html that can load and display all these information and copy it to target
    shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), os.path.join(target, "index.html"))

class TempCallback(tf.keras.callbacks.Callback):
  '''Callback that stores temporary results from the training process.
  Useful in case the system crashes - Can be used additionally to TensorBoard Callback.
  '''
  def __init__(self):
    pass
