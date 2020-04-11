# Argonaut-Examples

This folder contains various examples on how to use the argonaut system.

## Simple Multitask

Minimal example that shows how to run Argonaut with a single line of code on a configuration file.

The example trains a 4 layer convolutional network on the MNIST dataset. However, the classes of the dataset (10 overall) are randomly divided into 3 subsets, which are trained in sequence.

It trains each task for 2 epochs with a batch-size of 32 using SGD with a learning-rate of 0.001 and momentum of 0.9. (Note: these are just test parameters).

## Simple Model

Demonstrates how to use a simple keras model in the framework.
The model here is just a simple dense network that is directly injected into the argo `run_experiment` call.

## Custom Model

Demonstrates how to use a set of custom Models which are then selected by name from the config file.
The models live in a different file (i.e. `models.py`) and can be referenced from the config file by name.

> Note that you are not confined to a single file, but can also create an entire module (with submodules and everything) and load the model from there.
> However, keep in mind to provide the full path of the model class relative to the root model that you specified.

## Custom Experiment

Demonstrates how to setup a custom experiment pipeline (e.g. adds additional callbacks or data pre-processing) and run them.
This mechanism works similar to a custom model namespace in that you can define any module and then reference the name of the Experiment directly in the `run_experiment` call.