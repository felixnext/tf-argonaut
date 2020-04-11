'''
Example to create a custom list of experiments and load them with config.
'''

import argonaut as argo
from . import experiments


argo.run_experiment("CustomExperiment", "experiment.json", name="CustomExperiment", experiment_namespace=experiments)
