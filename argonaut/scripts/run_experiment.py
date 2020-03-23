'''
Execution file for experiments.
'''


import fire
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hormones import experiments as ex


def run_experiment(exp, config, location="../exp_logs", name=None, **args):
  '''Creates the experiment and runs it.

  Args:
    exp (str): Class name of the experiment as defined in `hormones.experiments` (e.g. 'Baseline')
    name (str): Actual name of the experiment
    model (str): Name of the model to use
    config (str): Name of the config file to use
    location (str): Path to the location of the config files used
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
    ex_cls = getattr(ex, exp)
  except:
    raise ValueError("Could not find the experiment with name ({})".format(exp))
  # check if config exists
  if not os.path.exists(config):
    raise ValueError("Could not find the config file ({})".format(config))

  # generate the experiment
  experiment = ex_cls.load(config, name=name, location=location, **args)

  # execute
  experiment.fit()

  # output summary
  print(str(experiment.summary))


if __name__ == '__main__':
  fire.Fire(run_experiment)
