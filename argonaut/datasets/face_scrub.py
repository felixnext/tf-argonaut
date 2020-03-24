'''
Loads the FaceScrub Datasets.

Note: Code for downloading the dataset is adapted from here: https://github.com/faceteam/facescrub

Paper: http://vintage.winklerbros.net/Publications/icip2014a.pdf
Website: http://vintage.winklerbros.net/facescrub.html
'''

import os
import glob
import cv2
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import hashlib
from tensorflow.keras import utils
from bunch import Bunch
from progressbar import ProgressBar
import urllib

from argonaut.utils import Summary
from argonaut.datasets import utils as ds_util


def face_scrub_download():
  '''Downloads the dataset to the specified folder.'''
  # download link files
  actors = "https://raw.githubusercontent.com/faceteam/facescrub/master/facescrub_actors.txt"
  actress = "https://raw.githubusercontent.com/faceteam/facescrub/master/facescrub_actresses.txt"
  files = [ds_util.download_dataset(f) for f in [actors, actress]]
  print(files)

  # iterate through files and download people
  folder = os.path.dirname(files[0])
  folder = os.path.join(folder, "facescrub")
  if not os.path.exists(folder):
    os.mkdir(folder)
  for data_file in files:
    df = pd.read_csv(data_file, sep="\t")
    # create progressbar
    bar = ProgressBar(max_value=len(df))
    # iterate through rows
    for index, row in df.iterrows():
      p_folder = os.path.join(folder, row.iloc[0])
      if not os.path.exists(p_folder):
        os.mkdir(p_folder)
      # download image
      fname = hashlib.sha1(row.iloc[3].encode('utf-8')).hexdigest() + '.jpg'
      img_file = os.path.join(p_folder, fname)
      if not os.path.exists(img_file):
        try:
          urllib.request.urlretrieve(row.iloc[3], img_file)
        except:
          print("A file download failed.")

      # update the progress bar
      bar.update(index)

  return folder

def _face_scrub_gen(folder, mode, size, pad_mode, num_classes, one_hot, seed):
  '''Creates a generator from the given image_list.

  Args:
    df_images (DataFrame): Dataframe that contains at least `class_id` and `path`
    size (tuple): Size the images should be scaled to
    pad_mode (str): See `resize_and_pad` function
  '''
  # create one_hot labels
  classes = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]
  num_classes = len(classes)
  eye = np.eye(num_classes)
  rand = random.Random(seed)

  # iterate through image_list
  for idx, cl in enumerate(classes):
    imgs = glob.glob(os.path.join(folder, cl, "*.jpg"))
    for img_path in imgs:
      # do selection
      r = rand.random()
      if mode == "train" and r > 0.85:
        continue
      elif mode == "test" and r <= 0.85:
        continue

      # load the image
      img = cv2.imread(img_path, 1) / 255
      img, scale = ds_util.resize_and_pad(img, size, pad_mode)
      # load the label
      cls_lbl = idx
      if one_hot == True:
        cls_lbl = np.copy(eye[[cls_lbl]][0])

      # return
      yield img, cls_lbl

def face_scrub_generator(size=(100, 100), pad_mode='fit_center', one_hot=True, seed=None, summary=None):
  '''Loads the Caltech CUB-200 2011 Dataset as a generator.

  Args:
    folder (str): Location of the dataset
    size (tuple): Rescaling size of the images
    pad_mode (str): Defines the padding mode (see `utils.resize_and_pad`)
    summary (Summary): Summary that contains processing information
  '''
  # load the metadata
  folder = face_scrub_download()
  classes = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]
  num_classes = len(classes)

  # setup seed
  if seed is None:
    seed = np.random.randint(0, 1000)

  # write to summary
  summary.add("data loading", "loaded cub200 (generator) dataset", {"folder": folder, "size": size, "pad_mode": pad_mode})

  # create final bunch
  return Bunch({
    "train": lambda: _face_scrub_gen(folder, "train", size, pad_mode, num_classes, one_hot, seed),
    "test": lambda: _face_scrub_gen(folder, "test", size, pad_mode, num_classes, one_hot, seed),
    "type": "generator",
    "size": (size[0], size[1], 3),
    "num_classes": num_classes,
    "class_names": classes,
    "unique_classes": np.eye(num_classes) if one_hot == True else np.arange(num_classes)
  }), summary
