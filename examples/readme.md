# Argonaut-Examples

This folder contains various examples on how to use the argonaut system.

## Simple Multitask

Minimal example that shows how to run Argonaut with a single line of code on a configuration file.

The example trains a 4 layer convolutional network on the MNIST dataset. However, the classes of the dataset (10 overall) are randomly divided into 3 subsets, which are trained in sequence.

It trains each task for 2 epochs with a batch-size of 32 using SGD with a learning-rate of 0.001 and momentum of 0.9. (Note: these are just test parameters).

## Simple Model

Demonstrates how to use a simple keras model in the framework.

## Custom Model

Demonstrates how to use a set of custom Models which are then selected by name from the config file.