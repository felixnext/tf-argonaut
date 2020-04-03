'''Executor to run the training.'''

import argonaut as argo

# execute the experiment
argo.run_experiment("Baseline", "experiment.json", name="SimpleExample")
