'''
Defines the basic experiment structure.

author: Felix Geilert
'''

from abc import abstractmethod, ABCMeta
import os
import json
import warnings
#import numpy as np
import tensorflow as tf
from time import time
from argonaut.utils import Summary
from argonaut import models
from argonaut.datasets import Pipeline
from argonaut.components import heads
from argonaut import callbacks as hc


# to fix h5py bug in tf?
os.environ['TF2_BEHAVIOR'] = '1'


class Experiment(metaclass=ABCMeta):
  '''Baseclass for any experiment.

  Experiments contain the following elements:
  - Topology and Structure (e.g. VGG16 Model with Hormone Layers)
  - Hyperparameters (e.g. Model Size, Learning Rate, etc.)
  - Dataset / Tasks (e.g. Mnist + Permutations)
  - Training Structure (e.g. Number of Epochs per Task, Sequential Learning, etc.)

  This class should help to structure this elements and execute them in a repeatable manner.
  An experiment allows to store the entire configuration into a file, so that it can be loaded and repeated.

  Args:
    model_name (str): Name of the model from the `hormones.models` module (e.g. 'simple.ConvNet')
    model_args (dict): Dictionary of input paramters for the model (note this has to be complete)
    data_args (dict): List of dictionaries with setup of datasets
    train_args (dict): List of dicts with training tasks and configurations
    location (str): Location to store the experiment records (default: None)
    name (str): The name of the experiment (if None generated automatically)
  '''
  def __init__(self, model_name, model_args=None, data_args=None, train_args=None, location=None, warm_start_dir=None, add_params={}, name=None):
    # generate name and summary
    self.name = name if name is not None else 'experiment_{}'.format(int(time()))
    self.summary = Summary(self.name)

    # find experiment subfolder
    if location is None:
      location = "./"
    if not os.path.exists(location):
      os.makedirs(location)
    loc_name = self.name
    pos = 1
    while os.path.exists(os.path.join(location, loc_name)):
      print("WARNING: Location name ({}) already exists, counting up".format(loc_name))
      pos += 1
      loc_name = "{}_{}".format(self.name, pos)
    self.location = os.path.join(location, loc_name)
    os.makedirs(self.location)

    # generate the model class
    self.model_name = model_name
    model_class = models
    for name in model_name.split("."):
      model_class = getattr(model_class, name)
    self.model_class = model_class
    self.warm_start = warm_start_dir
    self.model_add_params = add_params
    # check and load arguments
    self.model_args = Experiment._check_model_args(model_args)
    self.data_args = Experiment._check_data_args(data_args)
    self.train_args = Experiment._check_train_args(train_args, self.data_args)

  @classmethod
  def _check_model_args(cls, args):
    '''Safty function to check model arguments.'''
    # FEAT: add model dependent checks here
    return args

  @classmethod
  def _check_data_args(cls, args):
    '''Safty function to check data arguments.'''
    # check if list
    if not isinstance(args, list):
      raise ValueError("Provided Data Arguments should be a list!")

    # iterate through data
    for i, ds in enumerate(args):
      if "id" not in ds:
        raise IOError("Item at position {} has no id".format(i))
      if "dataset" not in ds and "input" not in ds:
        raise IOError("Dataset ({}) should have either a start dataset (`dataset`) or inherit from a pervious dataset (`input`)".format(ds["id"]))

    return args

  @classmethod
  def _check_train_args(cls, args, data_args):
    '''Safty function to check training arguments.'''
    # check if list
    if not isinstance(args, list):
      raise ValueError("Provided Data Arguments should be a list!")

    # iterate through data
    for i, task in enumerate(args):
      if "id" not in task:
        raise IOError("Item at position {} has no id".format(i))
      if "dataset" not in task:
        raise IOError("Task ({}) has no dataset".format(task["id"]))
      # check if dataset exists
      found = False
      for ds in data_args:
        if task["dataset"].startswith(ds["id"]):
          found = True
          break
      if not found:
        raise IOError("Dataset ({}) for task ({}) does not exist.".format(task["dataset"], task["id"]))

      # check various additional parameters
      for p in ["optimizer"]:
        if p not in task:
          raise IOError("Task ({}) has no attribute `{}`".format(task["id"], p))

      # provide default values
      for p, d in [("stop_early", False)]:
        if p not in task:
          warnings.warn("`{}` for task ({}) not found. Assume `{}`".format(p, task["id"], str(d)))
          task[p] = d

    return args

  @classmethod
  def _load_json(cls, file):
    '''Loads the relevant json files.'''
    # safty: check if file exists and load
    if not os.path.exists(file):
      raise IOError("Provided file did not exists ({})".format(file))
    with open(file, "r") as handle:
      data = json.load(handle)

    # extract baselocation from file
    baseloc = os.path.dirname(file)

    # define merge method
    def merge(tmp, exp, partial=False):
      # if unequal type, return newer (i.e. exp)
      if type(tmp) != type(exp):
        return exp

      # try to match list elements by order
      # FEAT: implement ID matching for list elements if available?
      if isinstance(tmp, list):
        # check if same length
        if len(tmp) > len(exp) or (len(tmp) < len(exp) and partial is False):
          return exp
        # match all elements
        for i in range(len(tmp)):
          exp[i] = merge(tmp[i], exp[i])
        return exp

      # dict iterate through keys
      if isinstance(tmp, dict):
        # check for headers
        for key in exp.keys():
          if key in tmp:
            tmp[key] = merge(tmp[key], exp[key])
          else:
            tmp[key] = exp[key]
        return tmp

      # if not special types, return newer
      return exp

    # check for previous files
    base_json = {}
    if "templates" in data:
      for tmp_file in data["templates"]:
        tmp_json = Experiment._load_json(os.path.join(baseloc, tmp_file))
        # join the base template
        base_json = merge(base_json, tmp_json)

    # join with original one
    data = merge(base_json, data)

    return data

  @classmethod
  def load(cls, file, name=None, location=None):
    '''Loads the experiment configuration from the given file

    Note: As this function is defined in the baseclass, the user is responsible to choose the relevant

    Args:
      file (str): File to load from
      name (str): Optional deviating name (if None use stored name)

    Returns:
      Experiment Instance

    Raises:
      IOError: If file does not exists or is corrupted
    '''
    # load the json file
    data = Experiment._load_json(file)

    # check all relevant parameters
    if "model" not in data or "class" not in data["model"] or "params" not in data["model"]:
      raise IOError("Provided file ({}) does not contain correct model parameters")
    model_name = data["model"]["class"]
    warm_start = data["model"]["warm_start"] if "warm_start" in data["model"] else None
    add_params = data["model"]["settings"] if "settings" in data["model"] else {}
    # check and load arguments
    model_args = Experiment._check_model_args(data["model"]["params"])
    data_args = Experiment._check_data_args(data["data"])
    train_args = Experiment._check_train_args(data["training"], data_args)

    # check if location is provided
    if location is None and "location" in data:
      location = data["location"]

    # use params to create new experiment
    return cls(model_name, model_args, data_args, train_args, name=name, location=location, warm_start_dir=warm_start, add_params=add_params)

  def store(self, file=None):
    '''Stores the configuration of the experiment into a file.

    Args:
      file (str): File inside the provided location to store the configuration
    '''
    # generate the output file
    with open(os.path.join(self.location, "config.csv" if file is None else file), "w") as handle:
      data = {
        "model": {
          "class": self.model_name,
          "params": self.model_args,
          "settings": self.model_add_params
        },
        "data": self.data_args,
        "training": self.train_args
      }
      # store
      json.dump(data, handle)

  def _eval(self, tasks):
    '''Evaluates the model on all given tasks.

    Note: This is used to evaluate the model in between epochs to measure forgetting

    Args:
      tasks (list): List of Task dicts loaded from config
    '''
    results = {}
    for task in tasks:
      # eval data
      dataset = self.datasets[task["dataset"]]
      # create inputs
      inputs = self._build_inputs(dataset, "default")

      # build the model
      head_model = getattr(heads, task["head"])(dataset, name=task["id"])
      model_out, aux_losses = self.build(inputs, task, head_model, False, "default")
      loss = getattr(tf.keras.losses, task["loss"])
      metrics = [] if "metrics" not in task else task["metrics"]

      # compile and load
      model = tf.keras.Model(inputs=inputs, outputs=model_out)
      model.compile(loss=loss, metrics=metrics)
      self.model.load_weights(os.path.join(self.location, "embedding.h5"))
      head_model.load_weights(os.path.join(self.location, "head_{}.h5".format(task["id"])))

      # eval the model
      if "type" in dataset and dataset.type == "tfdata":
        res = model.evaluate(dataset.test.batch(task["batch_size"]), verbose=0)
      else:
        res = model.evaluate(dataset.test.x, dataset.test.y, batch_size=task["batch_size"], verbose=0)
      results[task["id"]] = res

    return results

  def _build_inputs(self, dataset, mode="default"):
    # create relevant heads
    inputs = [tf.keras.Input(shape=dataset.size)]
    # check which inputs are actually provided
    if "has_complexity" in dataset and dataset.has_complexity is True:
      inputs.append(tf.keras.Input(shape=()))
    if "has_context" in dataset and dataset.has_context is True:
      context_size = 100 if "context_size" not in self.model_add_params else self.model_add_params["context_size"]
      inputs.append(tf.keras.Input(shape=(context_size,)))
    if mode == "emb_inp":
      # retrieve the embedding size
      emb_dim = 10
      if "emb_dim" in self.model_args:
        self.model_args["emb_dim"]
      # create relevant input
      inputs.append(tf.keras.Input(shape=(emb_dim,)))
    return inputs

  def _build_model(self, dataset, task, training=None, mode="default", task_id=None):
    '''Builds a model for a specific task.

    Args:
      dataset (dict): information about the dataset
      task (dict): Information about the task
      use_softmax (bool): Defines if the head should use softmax
      training (bool): defines if the model is in training mode
      mode (str): Build mode of the network (options are: [default, emb, emb_inp, logits])
    '''
    # create inputs
    inputs = self._build_inputs(dataset, mode)

    # retrieve the relevant head function for the network
    head_fn = getattr(heads, task["head"])(dataset, name=task["id"])

    # create the model according to specs
    model_out, aux_losses = self.build(inputs, task, head_fn, training, mode, dataset, task_id)
    if aux_losses is None:
      aux_losses = []

    # create the final model
    model = tf.keras.Model(inputs=inputs, outputs=model_out, name="model_{}".format(self.name))

    # define combined loss
    @tf.function
    def loss_fn(y_true, y_pred):
      # retrieve the actual loss function
      if hasattr(tf.keras.losses, task["loss"]) is False:
        raise ValueError("Could not find loss function ({}) in keras for task ({})!".format(task["loss"], task["id"]))
      loss = getattr(tf.keras.losses, task["loss"])(y_true, y_pred)

      # compute all auxiliary losses
      for aux_loss in aux_losses:
        loss += aux_loss(y_true, y_pred)
      return loss

    # output final combination
    return model, head_fn, loss_fn

  def _load_optimizer(self, task):
    '''Loads the optimizer for the given task.'''
    # define the optimizer
    if "optimizer" not in task:
      raise ValueError("Task ({}) should have an optimizer attribute".format(task["id"]))
    opt_params = task["optimizer"]["params"]
    if "lr" in task:
      if task["lr"]["type"] == "class":
        scheduler = getattr(tf.keras.optimizers.schedules, task["lr"]["class"])
        opt_params["learning_rate"] = scheduler(**task["lr"]["params"])
        # write summary
        self.summary.add("scheduler", "Added learning rate scheduler ({})".format(task["lr"]["class"]), task["lr"]["params"])
    opt = getattr(tf.keras.optimizers, task["optimizer"]["class"])(**opt_params)

    return opt

  def _compile_model(self, model, head_model, task, losses, warm_start):
    '''Creates the predefined optimizer and compile the model.

    Note: this is abstracted as certain approaches (e.g. HAT) operate on the gradients directly and might therefore want to use the gradient approach.

    Args:
      model (Model): The entire model that is inferenced
      head_model (Model): Sub model of the classification head
      task (dict): Dict of the current task
      losses (list): List of loss tensors from the current model (usually 1)
      warm_start (str): Path to warm_start location or None
    '''
    # NOTE: update weights
    opt = self._load_optimizer(task)

    # compile the model
    metrics = [] if "metrics" not in task else task["metrics"]
    print("METRICS: {}".format(metrics))
    model.compile(optimizer=opt, loss=losses, metrics=metrics)

    # load model (make sure that this is checkpoint path)
    if warm_start is not None:
      if warm_start.endswith(".h5"):
        model.load_weights(warm_start)
      else:
        # CHECK: check if this works (or we are just working on new weights..)
        self.model.load_weights(os.path.join(warm_start, "embedding.h5"))
        if os.path.exists(os.path.join(warm_start, "head_{}.h5".format(task["id"]))):
          head_model.load_weights(os.path.join(warm_start, "head_{}.h5".format(task["id"])))

    return model, head_model

  def fit(self, location=None):
    '''This function will build the model, load the relevant data and start the training process as defined in `train`.

    This function follows the main algorithm outlined in the paper

    Args:
      location (str): Location to store the experiment data
    '''
    # set location default if not provided
    if location is None:
      location = self.location

    # Initial building process and loading of the data
    self.model = self.model_class(**self.model_args)

    # load the relevant data
    self.dataset_pipe = Pipeline(self.data_args, self.train_args)
    self.datasets, self.summary = self.dataset_pipe.transform(summary=self.summary)

    # setup some meta-params
    warm_start = self.warm_start
    histories = {}
    epoch = 0
    model = None

    # iterate through datasets and train the model sequentially
    for i, task in enumerate(self.train_args):
      # retrieve task items
      dataset = self.datasets[task["dataset"]]
      # check warm load
      if i > 0:
        # NOTE: retrieve warm_start from previous task
        warm_start = self.location
      # generate task_id and folder
      id = task["id"]
      task_folder = os.path.join(self.location, "task_{}_{}".format(i, id))

      # debug: print name
      print("")
      print("-" * 40)
      print("TASK: {}".format(task["id"]))
      print("-" * 40)
      print("")

      # create the directories
      os.makedirs(os.path.join(task_folder, "checkpoints"))

      # create relevant heads
      model, head_model, losses = self._build_model(dataset, task, True, True, i)

      # check if model should be compiled
      model, head_model = self._compile_model(model, head_model, task, losses, warm_start)

      # create summary writer (allows models to write additional log information)
      log_dir = os.path.join(self.location, "log", task["id"])
      os.makedirs(os.path.join(log_dir, "metrics"))
      file_writer = tf.summary.create_file_writer(os.path.join(log_dir, "metrics"))
      file_writer.set_as_default()

      # create callbacks
      # NOTE: Update regarding parameters for callbacks (loss names?)
      callbacks = [tf.keras.callbacks.ModelCheckpoint(os.path.abspath(os.path.join(task_folder, "checkpoints")), save_weights_only=True)]
      # check early stopping
      if "stop_early" in task and task["stop_early"]["active"] is True:
        print("INFO: Activating early stopping")
        se = task["stop_early"]
        callbacks += [tf.keras.callbacks.EarlyStopping(monitor=se["item"], min_delta=se["min_delta"], patience=se["patience"])]
      # check tensorboard (NOTE: add summaries)
      if "tensorboard" in task and "enable" in task["tensorboard"] and task["tensorboard"]["enable"] is True:
        histo = 1 if "histogram" in task["tensorboard"] and task["tensorboard"]["histogram"] is True else 0
        callbacks += [tf.keras.callbacks.TensorBoard(log_dir=os.path.abspath(log_dir), histogram_freq=histo)]
      # check learning rate
      if "lr" in task:
        params = task["lr"]["params"]
        # check for different types
        if task["lr"]["type"] == "cyclic":
          callbacks += [hc.CyclicLR(**params)]
        elif task["lr"]["type"] == "epochstep":
          callbacks += [hc.EpochStepLR(start_epoch=epoch, max_epoch=task["epochs"], **params)]
        elif task["lr"]["type"] == "plateau":
          callbacks += [hc.PlateauLR(**params)]
        # check for all available types to avoid errors
        if task["lr"]["type"] not in ['cyclic', 'plateau', 'class', 'epochstep']:
          raise ValueError("Unkown learning rate type ({})".format(task["lr"]["type"]))
      # integrate mutli task validation after each epoch...
      callbacks += [hc.MultiTaskValidation(self.model, self.train_args[:(i + 1)], self.datasets, self.location, lambda x: self._build_inputs(x, "default"))]

      # start the training (call experiment training)
      model, history, self.summary = self.train(model, dataset, callbacks, epoch, task, i, self.summary)

      # update epoch
      if len(history.history) > 0:
        epoch += len(history.history["loss"])
        histories[task["id"]] = history

      # add summary of the model
      str_sum = []
      model.summary(print_fn=lambda x: str_sum.append(x))
      self.summary.add("model summary", "\n".join(str_sum))

      # store the final model
      model.save_weights(os.path.join(task_folder, "model.h5"), overwrite=True)
      self.model.save_weights(os.path.join(self.location, "embedding.h5"), overwrite=True)
      head_model.save_weights(os.path.join(self.location, "head_{}.h5".format(task["id"])), overwrite=True)

      # test against all
      #eval_res = self._eval(self.train_args[:(i + 1)])
      #self.summary.add("preliminary eval", "eval results after {} tasks".format(i+1), eval_res)

      # store the model structure?
      #with open(os.path.join(self.location, task["id"], "model.json"), "w") as json_file:
      #  json_file.write(model.to_json())

      # clear the session
      del head_model
      tf.keras.backend.clear_session()

    # save the final model
    # CHECK: see if this works after clear session?
    self.model.save_weights(os.path.join(self.location, "model.h5"))

    # iterate through all datasets and validate for final performance
    results = self._eval(self.train_args)
    self.summary.add("final eval", "eval results for all tasks", results)

    # finalize the summary
    self.summary.generate_summary(self.location, self.model, histories)
    self.store()

  def build(self, inputs, task, head_fn, training, mode, dataset=None, task_id=None):
    '''Builds the actual model.

    Note: this is used to provide custom model implementations

    Args:
      inputs (list): List of input vars
      dataset (Dataset): TF Dataset loaded from the data pipeline
      task (dict): Dictionary of the current task
      head_fn (Model): Sub-Model to create classification head
      is_training (bool): defines if the model is build for training or for inference

    Returns:
      output (Tensor): The output of the entire model
      losses (list): List of the losses from the model
    '''
    input = inputs[0]
    # build and compile model
    inner_out = self.model(input, training)
    final_out = head_fn(inner_out)

    return final_out, None

  @abstractmethod
  def train(self, model, dataset, callbacks, epoch, task_args, task_id, summary):
    '''Abstract function that defines the training structure in the child-classes.

    Note that this is the part of the experiment that will not be stored in file.
    Note that the model is already compiled with the optimizer provided in the settings.
    To change the optimizer, learning rate, etc. please use the corresponding callbacks

    Args:
      model (Model): Keras Model that should be fitted
      dataset: List of Bunch Datasets loaded
      callbacks (list): List of callbacks that are added to the train method
      epoch (int): Starting number of teh epoch (default 0)
      task_args (dict): Dict of settings for the current task as defined in config.json
      summary (Summary): Summary object to add results to

    Returns:
      model: Updated Model
      history: History of the training process
      summary: Updated Summary
    '''
    raise NotImplementedError()
