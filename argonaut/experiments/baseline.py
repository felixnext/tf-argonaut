'''
Defines the baseline experiment.

author: Felix Geilert
'''

#import tensorflow as tf
from .experiment import Experiment
from argonaut.callbacks import CurriculumCallback

class Baseline(Experiment):
  '''Defines the baseline experiment structure.'''
  def train(self, model, dataset, callbacks, epoch, task_args, task_id, summary):
    # select training based on dataset
    if "type" in dataset and dataset.type == "tfdata":
      # adds curriculum learning to the system (if set in params)
      if "curriculum" in task_args:
        params = task_args["curriculum"]
        cur_cb, dataset = CurriculumCallback.create(dataset, max_epochs=epoch + task_args["epochs"], start_epoch=epoch, filter=True, **params)
        callbacks.append(cur_cb)

      # check for shuffle
      ds_train = dataset.train
      if "shuffle" in task_args:
        ds_train = ds_train.shuffle(task_args["shuffle"])

      # fit the model
      history = model.fit(ds_train.batch(task_args["batch_size"]),
        epochs=epoch + task_args["epochs"], initial_epoch=epoch,
        validation_data=dataset.test.batch(task_args["batch_size"]),
        callbacks=callbacks)
    else:
      history = model.fit(dataset.train.x, dataset.train.y,
        batch_size=task_args["batch_size"],
        epochs=epoch + task_args['epochs'],
        validation_data=(dataset.test.x, dataset.test.y),
        callbacks=callbacks,
        initial_epoch=epoch)

    # write to summary
    hist = history.history
    hist_cp = {}
    for key in hist:
      hist_cp[key] = hist[key][-1]
    summary.add("fitted model", "fitted model to given dataset ({}) in task {}".format(task_args["dataset"], task_args["id"]), hist_cp)

    # results
    return model, history, summary
