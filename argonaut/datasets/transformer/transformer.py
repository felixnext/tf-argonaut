'''
Various utility functions for dataset loading.

author: Felix Geilert
'''

import os
from time import time
import cv2
import tensorflow as tf
from tensorflow import strings as tf_str
from abc import ABC, abstractmethod
import numpy as np
from bunch import Bunch
from argonaut.utils import Summary
from progressbar import ProgressBar
# sel-search relevant
from selectivesearch import selective_search
# entropy relevant
from skimage.filters.rank import entropy
from skimage.morphology import disk

class DSTransformer(ABC):
  @abstractmethod
  def _transform(self, dataset, summary, name):
    raise NotImplementedError()

  def transform(self, dataset, summary=None, name=None):
    '''Transforms the given dataset based in the defined operation.

    Args:
      dataset (bunch): Bunch of the loaded dataset
      summary (Summary): Summary class from hormones to write to
      name (str): Name of the operation to log in summary

    Returns:
      datasets (list): List of transformed bunch datasets
      summary (Summary): The updated summary
    '''
    # check if summary exists
    if summary is None:
      summary = Summary()

    return self._transform(dataset, summary, name)

class ExcludeTransformer(DSTransformer):
  '''Splits the dataset randomly in multiple subsets.
  According to datasets described in: https://arxiv.org/abs/1905.08077
  Default is 50-50 split in 2 datasets.

  Args:
    count (int): Number of resulting datasets
    split (list): List for the number of classes to use in respective dataset (if None split equally)
  '''
  def __init__(self, count=2, split=None):
    self.count = count
    self.split = split

  def _transform(self, dataset, summary=None, name=None):
    '''Retrieves a dataset and returns a list of datasets.'''
    # create split list for classes (mapping)
    cls_ls = dataset.unique_classes
    cls_map = np.tile(np.arange(self.count), int(np.ceil(len(cls_ls) / self.count)))[:len(cls_ls)]
    np.random.shuffle( cls_map )
    cls_ls = np.array([str(i) for i in cls_ls])
    cls_map = dict( zip(cls_ls, cls_map) )

    res = []

    # check how to handle the dataset type
    if "type" in dataset and dataset.type == "generator":
      # define function to handle generator
      def fnc(ds, pos):
        return lambda: [(yield x, y) for x, y in ds() if cls_map[str(y)] == pos]

      # iterate through potential generators
      for i in range(self.count):
        ds = Bunch(dataset.copy())
        ds.update(
          train=fnc(dataset.train, i),
          test=fnc(dataset.test, i),
          num_classes=sum([int(cls_map[k] == i) for k in cls_map]),
          class_names=[n for n, c in zip(dataset.class_names, dataset.unique_classes) if cls_map[str(c)] == i] if dataset.class_names is not None else None,
          unique_classes=np.array([c for c in dataset.unique_classes if cls_map[str(c)] == i])
        )
        res.append(ds)
    elif "type" in dataset and dataset.type == "preloaded":
      # create resulting datasets?
      items = []
      for i in range(self.count):
        items.append({'train': ([], []), 'test': ([], [])})

      # iterate through dataset
      for ds_type in ['train', 'test']:
        data = dataset[ds_type]
        for x, y in zip(data.x, data.y):
          # check the mapping of the class
          idx = cls_map[str(y)]
          # add data
          items[idx][ds_type][0].append(x)
          items[idx][ds_type][1].append(y)

      # convert to relevant bunches
      for i in range(len(items)):
        ds = Bunch(dataset.copy())
        ds.update(
          train=Bunch({'x': np.array(items[i]['train'][0]), 'y': np.array(items[i]['train'][1])}),
          test=Bunch({'x': np.array(items[i]['test'][0]), 'y': np.array(items[i]['test'][1])}),
          num_classes=np.unique(items[i]['train'][1], axis=0),
          unique_classes=np.array([c for c in dataset.unique_classes if cls_map[str(c)] == i])
        )
        res.append(ds)
    else:
      raise IOError("Operation not implemented for given dataset type (or type not available)")

    # return dataset list
    return res, summary

class OneHotTransformer(DSTransformer):
  '''Transforms class-ids to a one-hot mapping.'''
  def _transform(self, dataset, summary, name=None):
    if "type" in dataset and dataset.type == "generator":
      # define class mapping
      cls_map = dict(zip([str(x) for x in dataset.unique_classes], list(range(dataset.num_classes))))
      eye = np.eye(dataset.num_classes, dtype=np.float32)

      # add summary
      summary.add(name, 'Converted class_ids in dataset (generator) to one-hot', cls_map, 'one-hot')

      # iterate through all relevant datasets
      ds = Bunch(dataset.copy())
      ds.update(
        train=lambda: [(yield x, eye[cls_map[str(y)]]) for x, y in dataset.train()],
        test=lambda: [(yield x, eye[cls_map[str(y)]]) for x, y in dataset.test()],
        unique_classes=np.eye(dataset.num_classes)
      )
      return ds, summary
    else:
      raise IOError("Operation not implemented for the given dataset type (or type not available)")

class PermuteTransformer(DSTransformer):
  '''Generates a copy of the dataset where classes are permuted.
  According to datasets described in: https://arxiv.org/abs/1905.08077
  '''
  def __init__(self, one_hot=True):
    self.one_hot = one_hot

  def _transform(self, dataset, summary, name=None):
    '''Retrieves a dataset and returns a list of datasets.'''
    # retrieve single values
    cls_ls = dataset.unique_classes

    if "type" in dataset and dataset.type == "tfdata":
      # prepare data
      cls_ls2 = np.array(['["{}"]'.format(" ".join(str(x) for x in i)) for i in cls_ls])
      ids = np.arange(10)
      np.random.shuffle(ids)

      # create lookup tensors
      ten = tf.convert_to_tensor(cls_ls)
      table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(cls_ls2, ids), -1)

      # define permutation function
      def permute_fn(img, lbl):
        str_lbl = tf_str.as_string(tf.cast(lbl, tf.int32))
        str_lbl = [str_lbl[i] for i in range(str_lbl.shape[0])]
        id = table.lookup(tf_str.format("[{}]", tf_str.join(str_lbl, ' ')))
        lbl_perm = tf.slice(ten, tf.stack([id, 0], axis=-1), [1, dataset.num_classes])
        return img, tf.squeeze(lbl_perm)

      # permute the dataset
      ds = Bunch(dataset.copy())
      ds.update(
        train=dataset.train.map(permute_fn),
        test=dataset.test.map(permute_fn)
      )
      return ds, summary
    elif "type" in dataset and dataset.type == "generator":
      # data
      cls_ls2 = np.copy(cls_ls)
      cls_ls = np.array([str(i) for i in cls_ls])
      np.random.shuffle(cls_ls2)
      cls_map = dict( zip(cls_ls, list(cls_ls2)) )

      def _gen(dataset):
        for x, y in dataset:
          yield np.copy(x), cls_map[str(y)]

      # add to summary
      summary.add(name, 'Permute the dataset (generator) to create a new task', cls_map, 'permute')

      ds = Bunch(dataset.copy())
      ds.update(
        train=lambda: _gen(dataset.train()),
        test=lambda: _gen(dataset.test())
      )
      return ds, summary
    else:
      # data
      cls_ls2 = np.copy(cls_ls)
      cls_ls = np.array([str(i) for i in cls_ls])
      np.random.shuffle(cls_ls2)
      cls_map = dict( zip(cls_ls, list(cls_ls2)) )

      # add to summary
      summary.add(name, 'Permute the dataset to create a new task', cls_map, 'permute')

      ds = Bunch(dataset.copy())
      ds.update(
        train=Bunch({'x': np.copy(dataset.train.x), 'y': np.array([cls_map[str(k)] for k in dataset.train.y])}),
        test=Bunch({'x': np.copy(dataset.test.x), 'y': np.array([cls_map[str(k)] for k in dataset.test.y])}),
        num_classes=len(cls_ls)
      )
      return ds, summary

class TFDataTransformer(DSTransformer):
  '''Transforms a python generator to tfdata.

  Args:
    shuffle (int): The amount of items used for shuffling
  '''
  def __init__(self, cache=False, shuffle=None):
    self.shuffle = shuffle
    self.cache = cache

  def _transform(self, dataset, summary, name=None):
    # safty check
    if "type" not in dataset or dataset.type != "generator":
      raise ValueError("Expected dataset to be of type generator (found: ({}))!".format(dataset.type if "type" in dataset else "NONE"))

    # retrieve datatypes
    gen = dataset.train()
    out = next(gen)
    # check functions
    check_dtype = lambda x: tf.dtypes.as_dtype(x.dtype if isinstance(x, np.ndarray) else np.dtype(type(x)))
    check_shape = lambda x: list(x.shape) if hasattr(x, "shape") else []
    # special case logic to handle additional input signals (e.g. context signals or complexity)
    if len(out) > 2:
      inner_dtypes = [check_dtype(x) for x in out[:1] + out[2:]]
      inner_shapes = [check_shape(x) for x in out[:1] + out[2:]]
      dtypes = (tuple(inner_dtypes), check_dtype(out[1]))
      shapes = (tuple(inner_shapes), check_shape(out[1]))
      # update the generators to new order
      reorder = lambda tpl: (tuple(tpl[:1] + tpl[2:]), tpl[1])
      train_fct = lambda: [(yield reorder(tpl)) for tpl in dataset.train()]
      test_fct = lambda: [(yield reorder(tpl)) for tpl in dataset.test()]
    else:
      dtypes = tuple([check_dtype(x) for x in out])
      shapes = tuple([check_shape(x) for x in out])
      train_fct = dataset.train
      test_fct = dataset.test
    del gen

    print("INFO: Generated TF-Dataset with following dtypes: {}".format(dtypes))

    # create the datasets
    ds_train = tf.data.Dataset.from_generator(train_fct, output_types=dtypes, output_shapes=shapes)
    ds_test  = tf.data.Dataset.from_generator(test_fct, output_types=dtypes, output_shapes=shapes)

    # check if cache
    if self.cache is True:
      print("INFO: caching is activated")
      path = os.path.join("../tf_cache", "{}_{}".format(dataset.name, int(time())))
      if not os.path.exists(path):
        os.makedirs(path)
      ds_train = ds_train.cache(os.path.join(path, "train"))
      ds_test = ds_test.cache(os.path.join(path, "test"))

    # check for shuffle
    if self.shuffle is not None:
      ds_train = ds_train.shuffle(self.shuffle)
      ds_test  = ds_test.shuffle(self.shuffle)

    # add to summary
    summary.add("data loading", "converted dataset to tfdata", {"shuffle": self.shuffle, "dtypes": dtypes})

    # create bunch package
    ds = Bunch(dataset.copy())
    ds.update(
      train=ds_train,
      test=ds_test,
      type='tfdata'
    )
    return ds, summary

class ResizeTransformer(DSTransformer):
  '''Transformer for input images, that converts all images to a fixed size.

  Args:
    shape (tuple): Tuple of shape information
  '''
  def __init__(self, shape):
    self.shape = tuple(shape)

  def _resize(self, img):
    '''Internal resize function.'''
    return cv2.resize(img, self.shape)

  def _transform(self, dataset, summary, name=None):
    if "type" in dataset and dataset.type == "generator":
      # add summary
      summary.add(name, 'Resized all input images to fixed size', self.shape, 'img-resize')

      # iterate through all relevant datasets
      ds = Bunch(dataset.copy())
      ds.update(
        train=lambda: [(yield self._resize(x), y) for x, y in dataset.train()],
        test=lambda: [(yield self._resize(x), y) for x, y in dataset.test()],
        size=(self.shape[0], self.shape[1], dataset.size[2])
      )
      return ds, summary
    else:
      raise IOError("Operation not implemented for the given dataset type (or type not available)")

class TaskContextTransformer(DSTransformer):
  '''Transformer that uses the current task position to insert a new task input signal as one_hot vector.

  Args:
    dim_size (int): Size of the one_hot context vector
    pos (int): Position of the current task inside the vector
  '''
  def __init__(self, dim_size, pos):
    self.dim_size = dim_size
    self.pos = pos

  def _add_item(self, x, vector):
    if isinstance(x, tuple):
      return x + (vector,)
    if isinstance(x, list):
      return x + [vector]
    return (x, vector)

  def _transform(self, dataset, summary, name=None):
    if "type" in dataset and dataset.type == "generator":
      # add summary
      summary.add(name, 'Resized all input images to fixed size', self.shape, 'img-resize')

      # calculate the relevant vector
      vector = np.eye(self.dim_size, dtype=np.float32)[self.pos]

      # iterate through all relevant datasets
      ds = Bunch(dataset.copy())
      ds.update(
        train=lambda: [(yield self._add_item(x, vector), y) for x, y in dataset.train()],
        test=lambda: [(yield self._add_item(x, vector), y) for x, y in dataset.test()],
        has_context=True
      )
      return ds, summary
    else:
      raise IOError("Operation not implemented for the given dataset type (or type not available)")

class ComplexityTransformer(DSTransformer):
  '''Defines the complexity of the elements.

  This should make sure that at least one element lies at complexity zero.

  Note: Complexity of the elements are defined relative to the dataset
  Performance Note: It is recommended to call this dataset after exclusion of performation, as startup can take some time to identify max and min values
  '''
  def _complexity(self, img):
    raise NotImplementedError("Abstract Base Class")

  def _transform(self, dataset, summary, name=None):
    if "type" in dataset and dataset.type == "generator":
      # add summary
      summary.add(name, 'Added complexity to the Dataset', None, 'complexity')
      bar = ProgressBar()
      pos = 0

      # iterate through dataset to define min and max values
      max_val = {}
      min_val = {}
      train_get = dataset.train()
      for x, y in train_get:
        # calculate values
        c = self._complexity(x)
        pos += 1
        bar.update(pos)
        lbl = str(y)

        # check values
        if lbl not in min_val or min_val[lbl] > c:
          min_val[lbl] = c
        if lbl not in max_val or max_val[lbl] < c:
          max_val[lbl] = c
      # define normalization values
      lim_val = dict([(x, max(y - min_val[x], 0.001)) for x, y in max_val.items()])
      print("")

      # define normalization function
      def normalize(x, y):
        return np.cast[np.float32]((self._complexity(x) - min_val[str(y)]) / lim_val[str(y)])

      # iterate through all relevant datasets
      ds = Bunch(dataset.copy())
      ds.update(
        train=lambda: [(yield x, y, normalize(x, y)) for x, y in dataset.train()],
        test=lambda: [(yield x, y, normalize(x, y)) for x, y in dataset.test()],
        has_complexity=True
      )
      return ds, summary
    else:
      raise IOError("Operation not implemented for the given dataset type (or type not available)")

class FixedComplexityTransformer(ComplexityTransformer):
  '''Adds a fixed complexity to every element in the dataset.

  Args:
    value (float): value that should be applied as fixed complexity
  '''
  def __init__(self, value=0.):
    self.value = float(value)

  def _complexity(self, img):
    return self.value

class RandomComplexityTransformer(ComplexityTransformer):
  '''Defines a random complexity Transformation.

  Used as baseline and for testing to reduce wait-times.

  Args:
    seed (int): Randomization seed
  '''
  def __init__(self, seed=None):
    self.rand = np.random.RandomState(seed)

  def _complexity(self, img):
    return self.rand.rand()

class VarianceComplexityTransformer(ComplexityTransformer):
  '''Measures the complexity of elements by the variance inside the image.'''
  def _complexity(self, img):
    return np.var(img)

class EntropyComplexityTransformer(ComplexityTransformer):
  '''Measures the complexity of an Image through shannon entropy.

  Args:
    wnd_size (int): Size of the entropy window (should be in relation to image size)
    square (bool): If the entropy value should be squared (allows to increase high complex distances)
  '''
  def __init__(self, wnd_size=5, square=True):
    self.wnd = wnd_size
    self.square = square

  def _complexity(self, img):
    ent_img = img
    if len(ent_img.shape) > 2:
      ent_img = np.mean(ent_img, axis=-1)
    if np.max(ent_img) > 1:
      ent_img = ent_img / 255.
    ent = entropy(ent_img, disk(self.wnd))
    if self.square is True:
      ent = np.power(ent, 2)
    return np.mean(ent)

class SelSearchComplexityTransformer(ComplexityTransformer):
  '''Adds complexity through selective search into the dataset.

  Paper: https://link.springer.com/article/10.1007/s11263-013-0620-5

  Note that the threshold parameter is dependent on the image size

  Args:
    scale (float): Scale of the image to use (higher = larger clusters, i.e. coarser complexity)
    sigma (float): Width of gaussian kernel (higher = more items / higher complexity)
    min_size (int): Min size of clusters to be relevant (higher = coarser scan)
  '''
  def __init__(self, scale=50., sigma=0.5, min_size=20):
    self.scale = scale
    self.sigma = sigma
    self.min_size = min_size

  def _complexity(self, img):
    lbl, reg = selective_search(img, self.scale, self.sigma, self.min_size)
    return len(reg)

def exclude(dataset, count=2, split=None, summary=None, name=None):
  '''Splits the dataset randomly in multiple subsets.
  According to datasets described in: https://arxiv.org/abs/1905.08077
  Default is 50-50 split in 2 datasets.

  Args:
    dataset (bunch): Already loaded dataset structure
    count (int): Number of resulting datasets
    split (list): List for the number of classes to use in respective dataset (if None split equally)
    summary (Summary): Summary item to insert the operation
    name (str): Name of the operation (default: exclude)

  Returns:
    ls: Array of bunch datasets that match the elements
    sum: Summary of operations executed
  '''
  # create split list for classes (mapping)
  cls_ls = np.unique(dataset.train.y, axis=0)
  cls_map = np.tile(np.arange(count), int(np.ceil(len(cls_ls) / count)))[:len(cls_ls)]
  np.random.shuffle( cls_map )
  cls_ls = np.array([str(i) for i in cls_ls])
  cls_map = dict( zip(cls_ls, cls_map) )

  # write info to summary
  if summary is not None:
    summary.add(name, 'Split the dataset in subset', cls_map, 'exclude')

  # check which type of dataset
  if "type" in dataset and dataset.type == "tfdata":
    # define interal function
    def exclude_fn(image, label):
      return image, label

    # simply return function
    return exclude_fn
  else:
    # create resulting datasets?
    items = []
    for i in range(count):
      items.append({'train': ([], []), 'test': ([], [])})

    # iterate through dataset
    for ds_type in ['train', 'test']:
      data = dataset[ds_type]
      for x, y in zip(data.x, data.y):
        # check the mapping of the class
        idx = cls_map[str(y)]
        # add data
        items[idx][ds_type][0].append(x)
        items[idx][ds_type][1].append(y)

    # convert to relevant bunches
    res = []
    for i in range(len(items)):
      res.append(Bunch({
        'test': Bunch({'x': np.array(items[i]['test'][0]), 'y': np.array(items[i]['test'][1])}),
        'train': Bunch({'x': np.array(items[i]['train'][0]), 'y': np.array(items[i]['train'][1])}),
        'size': dataset.size,
        'num_classes': np.unique(items[i]['train'][1], axis=0),
        'class_names': dataset.class_names,
      }))

    return res, summary

def permute(dataset, summary=None, name=None):
  '''Generates a copy of the dataset where classes are permuted.
  According to datasets described in: https://arxiv.org/abs/1905.08077

  Args:
    dataset (bunch): Already loaded dataset structure
    summary (Summary): Summary item to insert the operation
    name (str): Name of the operation (default: exclude)

  Returns:
    Bunch structure of permuted dataset
  '''
  # check if summary exists
  if summary is None:
    summary = Summary()

  # retrieve single values
  cls_ls = np.unique(dataset.train.y, axis=0)
  cls_ls2 = np.copy(cls_ls)
  np.random.shuffle(cls_ls2)
  cls_ls = np.array([str(i) for i in cls_ls])
  cls_map = dict( zip(cls_ls, list(cls_ls2)) )

  # add to summary
  if summary is not None:
    summary.add(name, 'Permute the dataset to create a new task', cls_map, 'permute')

  return Bunch({
    'test': Bunch({'x': np.copy(dataset.test.x), 'y': np.array([cls_map[str(k)] for k in dataset.test.y])}),
    'train': Bunch({'x': np.copy(dataset.train.x), 'y': np.array([cls_map[str(k)] for k in dataset.train.y])}),
    'size': dataset.size,
    'num_classes': len(cls_ls),
    'class_names': dataset.class_names,
  }), summary

def permute_list(dataset, count=2, summary=None, name=None):
  '''Creates a permuted list of elements.

  Args:
    dataset (bunch): Dataset to permute
    count (int): Number of permutations (first is original dataset)
    summary (Summary): Summary item to insert the operation
    name (str): Name of the operation (default: exclude)

  Returns:
    res: List of bunch datasets that contain the permutations
    summary: modified summary
  '''
  res = [dataset]
  for i in range(count - 1):
    perm, summary = permute(dataset, summary, name)
    res.append(perm)
  return res, summary
