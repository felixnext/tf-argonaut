'''
Example that explains how to use a set of custom models for experiments.
'''

import argonaut as argo
from . import models

argo.run_experiment("Baseline", "experiment.json", name="CustomModel", model_namespace=models)
