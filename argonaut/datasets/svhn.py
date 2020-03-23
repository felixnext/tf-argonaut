'''
Loads the SVHN Datasets.

Note: this uses the 2011 version

Paper: http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf
Website: http://ufldl.stanford.edu/housenumbers/
'''

import os
import cv2
from scipy.io import loadmat
import glob
import random
import numpy as np
from bunch import Bunch
from progressbar import ProgressBar

#from hormones.utils import Summary
from hormones.datasets import utils as ds_util


def svhn_download():
  '''Downloads the dataset to the specified folder.'''
  # try to load data
  train_mat = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
  test_mat = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
  folders = []
  for link in [train_mat, test_mat]:
    file_path = ds_util.download_dataset(link)
    # create a folder
    folder_path = os.path.join(os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0])
    if not os.path.exists(folder_path):
      os.mkdir(folder_path)
    # extrat mat using scipy?
    data = loadmat(file_path, variable_names=['X', 'y'], appendmat=True)
    images = data.get('X')
    labels = data.get('y')
    bar = ProgressBar(max_value=images.shape[-1])
    # iterate through images
    for i in range(images.shape[-1]):
      # load data
      img = images[..., i]
      lbl = labels[i][0]
      # check class folder
      cls_path = os.path.join(folder_path, str(lbl))
      if not os.path.exists(cls_path):
        os.mkdir(cls_path)
      img_path = os.path.join(cls_path, "img_{}.jpg".format(i))
      if not os.path.exists(img_path):
        cv2.imwrite(img_path, img)
      bar.update(i)
    folders.append(folder_path)
  train_folder, test_folder = folders
  return train_folder, test_folder

def _svhn_gen(folder, num_classes, one_hot, seed):
  '''Creates a generator from the given image_list.

  Args:
    folder (str): folder path of the relevant data
  '''
  classes = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]
  eye = np.eye(len(classes))

  # create list of all data
  datapoints = []

  # iterate through relevant classes
  for i, c in enumerate(classes):
    imgs = glob.glob(os.path.join(folder, c, "*.jpg"))
    for img_path in imgs:
      datapoints.append((img_path, c))

  # shuffle the dataset list
  random.Random(seed).shuffle(datapoints)
  #shuffle(datapoints)

  # iterate data
  for img_path, c in datapoints:
    # load the image
    img = cv2.imread(img_path, 1)
    lbl = classes.index(c)
    if img is None:
      continue
    img = img.astype("float32")

    # update the image
    img /= 255.0
    # update label
    if one_hot == True:
      lbl = np.copy(eye[[lbl]][0])

    yield img, lbl

def svhn_generator(one_hot=True, seed=None, summary=None):
  '''Loads the Caltech CUB-200 2011 Dataset as a generator.

  Args:
    folder (str): Location of the dataset
    size (tuple): Rescaling size of the images
    pad_mode (str): Defines the padding mode (see `utils.resize_and_pad`)
    summary (Summary): Summary that contains processing information
  '''
  # load the metadata
  train_folder, test_folder = svhn_download()
  num_classes = 10

  # setup seed
  if seed is None:
    seed = np.random.randint(0, 1000)

  # write to summary
  summary.add("data loading", "loaded cub200 (generator) dataset", {"test_folder": test_folder, "train_folder": train_folder, "one_hot": one_hot})

  # create final bunch
  return Bunch({
    'name': 'svhn',
    "train": lambda: _svhn_gen(train_folder, num_classes, one_hot, seed),
    "test": lambda: _svhn_gen(test_folder, num_classes, one_hot, seed),
    "type": "generator",
    "size": (32, 32, 3),
    "num_classes": num_classes,
    "class_names": None,
    "unique_classes": np.eye(num_classes) if one_hot == True else np.arange(num_classes)
  }), summary
