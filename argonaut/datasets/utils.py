'''
Defines various data functions that are used during loading.
'''

import os
import urllib
from progressbar import ProgressBar
from zipfile import ZipFile
import tarfile
import warnings
import cv2
import numpy as np


def _resize(img, size):
  img_size = img.shape
  if type(size) == tuple or type(size) == list or type(size) == np.ndarray:
    frac = min((size[0] / img_size[0], size[1] / img_size[1]))
    scale = (frac, frac)
  elif type(size) == int:
    frac = float(size) / max(img_size)
    scale = (frac, frac)
  else:
    raise ValueError("Size has unkown type ({}: {})".format(type(size), size))

  return cv2.resize(img, None, fx=scale[0], fy=scale[1]), scale

def _pad(img, color_arr, pos):
  if pos == "topleft":
    pad_pos = [0, 0]
  elif pos == "center":
    pad_pos = [int(np.floor((color_arr.shape[0] - img.shape[0]) / 2)), int(np.floor((color_arr.shape[1] - img.shape[1]) / 2))]

  # check additional padding modes
  color_arr[pad_pos[0]:pad_pos[0] + img.shape[0], pad_pos[1]:pad_pos[1] + img.shape[1], :] = img
  return color_arr

def resize_and_pad(img, size, pad_mode, color=None):
  '''Rescales the image and pads the remaining stuff.

  Relevant pad-modes:
  - stretch = stretches the image to the new aspect ratio
  - move_center = centers the image and fills the rest with `color` (or black)
  - move_center_random = centers the image and fills the rest with random color
  - move_topleft = adds image to the top left and fills rest with color
  - move_topleft_random
  - fit_center
  - fit_center_random
  - fit_topleft
  - fit_topleft_random

  Args:
    img (numpy.array): Array containing the image data.
    size (tuple): target size of the image
    pad_mode (str): Mode used for padding the image

  Returns:
    img (numpy.array): The updated image
    scale (tuple): The stretch factors along x and y axis (for adjustment of labels)
  '''
  # simply resize the image
  if pad_mode == 'stretch':
    return cv2.resize(img, size), (size[0] / img.shape[0], size[1] / img.shape[1])

  # create the padding array
  # NOTE: check for data type (0 to 1 vs 0 to 255) (int vs float)
  pad_split = pad_mode.split('_')
  color_arr = None
  if len(pad_split) <= 2 or pad_split[2] != 'random':
    color = [0, 0, 0] if color is None else color
    color_arr = np.stack([np.full(size, c, dtype=np.float32) for c in color], axis=-1)
  else:
    color_arr = np.random.rand(*size, img.shape[-1] if len(img.shape) > 2 else 1)

  # FEAT: add offset to the border of the image (positive and negative)

  # check if only move
  if pad_split[0] == 'move':
    # check if size is at end
    scale = [1, 1]
    if size[0] < img.shape[0] or size[1] < img.shape[1]:
      img, scale = _resize(img, size)
    # update the final image
    img = _pad(img, color_arr, pad_split[1])

    return img, scale

  if pad_split[0] == 'fit':
    # find most relevant side to use
    img, scale = _resize(img, size)
    # update the final image
    img = _pad(img, color_arr, pad_split[1])

    return img, scale

  raise ValueError("resize mode ({}) not found!".format(pad_split[0]))

def _file_path(cache_dir=None, create=False):
  '''Generates the full file-path based on the given name.'''
  # update the directory based on the current path
  if cache_dir is None:
    cache_dir = os.path.join(os.path.expanduser('~'), '.hormones')
  cache_path = os.path.expanduser(cache_dir)
  # attempt to create directory
  if not os.path.exists(cache_path):
    if create == True:
      os.makedirs(cache_path)
    else:
      warnings.warn("Could not create cache directory!")
  # check additional access rights
  if not os.access(cache_path, os.W_OK):
    warnings.warn("No access to default data dir: {} - using /tmp instead".format(cache_path))
    cache_path = os.path.join('/tmp', '.hormones')

  return cache_path

class DownloadBar():
  def __init__(self):
    self.pbar = None

  def __call__(self, block_num, block_size, total_size):
    if not self.pbar:
      self.pbar = ProgressBar(maxval=total_size)
      self.pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
      self.pbar.update(downloaded)
    else:
      self.pbar.finish()

def download_dataset(link, cache_dir=None):
  '''Downloads the relevant dataset and extracts it.
  Args:
    name (str): Name of the model to download (options are: [twitter, wikipedia])
  Returns:
    True if successful, otherwise False
  '''
  # retrieve file and path
  folder = _file_path(cache_dir, create=True)
  target_file = link[link.rfind("/") + 1:]
  file = os.path.join(folder, target_file)
  target_folder = os.path.join(folder, os.path.splitext(target_file)[0])

  # check if files exists otherwise download
  if os.path.isfile(file):
    print('File found, no download needed')
  else:
    try:
      urllib.request.urlretrieve(link, file, DownloadBar())
    except:
      raise IOError("download of {} data failed".format(link))

  # check if extract exists
  if os.path.isdir(target_folder):
    print('File is already extracted')
  else:
    try:
      ext = os.path.splitext(target_file)[1].lower()
      if ext == ".zip":
        # Create a ZipFile Object and load sample.zip in it
        with ZipFile(file, 'r') as zipObj:
          # Extract all the contents of zip file in current directory
          zipObj.extractall(target_folder)
      elif ext in [".tar", '.gz', '.tgz']:
        mode = ""
        if ext == ".tgz" or ext == ".gz":
          mode = "gz"
        tar = tarfile.open(file, 'r:{}'.format(mode))
        tar.extractall(target_folder)
        tar.close()
      else:
        target_folder = target_folder + ext
    except:
      raise IOError("extraction of {} data failed".format(file))

  return target_folder
