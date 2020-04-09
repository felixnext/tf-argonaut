'''
Defines execution functions that can easily be called.

'''

import os
from argonaut import experiments as ex

def run_experiment(exp, config, location="../exp_logs", name=None, experiment_namespace=None, model=None, model_namespace=None, **args):
  '''Creates the experiment and runs it.

  Args:
    exp (str): Class name of the experiment as defined in `hormones.experiments` (e.g. 'Baseline')
    name (str): Actual name of the experiment
    model (str): Name of the model to use
    config (str): Name of the config file to use
    location (str): Path to the location of the config files used
    name (str): Name of the experiment (will replace default name)
    experiment_namespace (module): Python Module from which experiments might be loaded (if not found default to internal experiments)
    model (tf.keras.Model): Model that will be loaded (instead the one specified in arguments)
    model_namespace (module): Python Module from which experiments can be loaded alternatively (if not found in internal experiments)
    args (dict): Additional parameters that should be overwritten from config
  '''
  # retrieve name from data
  if name is None:
    name = os.path.splitext(os.path.basename(config))[0]

  # print output
  print("")
  print("=" * 50)
  print("EXPERIMENT: {}".format(name))
  print("=" * 50)
  print("")

  ex_cls = None
  # iterate through possible options
  try:
    try:
        ex_cls = getattr(experiment_namespace, exp)
    except:
        ex_cls = getattr(ex, exp)
  except:
    raise ValueError("Could not find the experiment with name ({})".format(exp))

  # check if config exists
  if not os.path.exists(config):
    raise ValueError("Could not find the config file ({})".format(config))

  # generate the experiment
  experiment = ex_cls.load(config, name=name, location=location, model=model, model_namespace=model_namespace, **args)

  # execute
  experiment.fit()

  # output summary
  print(str(experiment.summary))
