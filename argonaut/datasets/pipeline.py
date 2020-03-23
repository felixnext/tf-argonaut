'''
Defines a datapipeline that can be constructed from string.

author: Felix Geilert
'''

from hormones import datasets as ds_mod
from hormones.utils import Summary

class Pipeline():
  '''Loads a data pipeline based on string operations and use data for the results.

  In general these pipelines have 3 types of operations:
  1. loading - load data from file and generate a dataset
  2. modify - Modify the datasource (e.g. augmentation or permutation)
  3. modify and copy - Same as modify, but the operations are applied to a copy of the dataset

  Output data types can be of 2 categories:
  - loaded - data stored directly in RAM
  - generated - data stored in generator that is loaded at runtime (for larger datasets)

  Args:
    data_args (dict): Dict that contains relevant arguments for the datasets to load
    train_args (dict): Dict that contains information about the training process (i.e. which datasets are relevant to keep)
    clean (bool): defines if unused datasets should be discarded
  '''
  def __init__(self, data_args, train_args, clean=True):
    # store the arguments
    self.args = data_args
    self.clean = clean
    # iterate to find relevant datasets
    used_ds = set()
    for task in train_args:
      used_ds.add(task["dataset"])
    self.datasets = used_ds

  def transform(self, summary=None):
    '''Executes the data loading and transformation.'''
    if summary is None:
      summary = Summary()
    datasets = {}
    # iterate through all sets
    for ds_args in self.args:
      # load the relevan dataset
      ds = None
      if "input" in ds_args:
        if ds_args["input"] not in datasets:
          raise LookupError("Could not find the input ({}) for dataset ({}). Please check execution order! Note that the system might append `_X` to the name in case the function returns multiple results.".format(ds_args["input"], ds_args["id"]))
        ds = datasets[ds_args["input"]]
      else:
        # load dataset
        ds_class = ds_mod
        for name in ds_args["dataset"].split("."):
          ds_class = getattr(ds_class, name)
        # init dataset
        params = {}
        if "params" in ds_args:
          params = ds_args["params"]
        ds, summary = ds_class(**params, summary=summary)
      # FEAT: avoid loading of dataset filtered by curriculum (rebuild pipeline with better curriculum filtering from base functions on)

      # check operations
      if "operations" in ds_args:
        for op in ds_args["operations"]:
          # NOTE: might change the utils to a more general module?
          tf_cls = getattr(ds_mod.transformer, op["name"])
          params = {}
          if "params" in op:
            params = op["params"]
          tf = tf_cls(**params)
          # handle array outputs for chained results
          if isinstance(ds, list):
            ds_arr = []
            for ds_i in ds:
              ds_out, summary = tf.transform(ds_i, summary=summary)
              ds_arr.append(ds_out)
            ds = ds_arr
          else:
            ds, summary = tf.transform(ds, summary=summary)

      # add completed data
      if isinstance(ds, list):
        for i, ids in enumerate(ds):
          datasets["{}_{}".format(ds_args["id"], i)] = ids
      else:
        datasets[ds_args["id"]] = ds

    if self.clean is False:
      return datasets, summary

    # check which datasets should be kept
    res = {}
    for key in datasets:
      if key in self.datasets:
        res[key] = datasets[key]
    del datasets

    return res, summary
