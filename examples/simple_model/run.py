'''Define a simple model and pass it to the '''

import argonaut as argo
import tensorflow as tf
from tensorflow import keras

# -- define the model --
# note that the input has to relate to the input of the dataset (in this case 48x48 scaled MNIST data with 3 channels)
inputs = keras.Input(shape=(48, 48, 3))
x = keras.layers.Reshape((48 * 48 * 3,))(inputs)
x = keras.layers.Dense(40, activation=tf.nn.relu)(x)
# note that the output is not the final layer of the model
# - the framework will automatically add a head for the specific datasource (number of classes etc) to the model (in thise case `ClassifyHeadDense`)
# - you should therefore not apply final activations (e.g. softmax) on top
outputs = keras.layers.Dense(100, activation=tf.nn.relu)(x)
# note that you can define your own heads if required (see `custom_model` example)
model = keras.Model(inputs=inputs, outputs=outputs)

# -- execute the experiment --
argo.run_experiment("Baseline", "experiment.json", name="SimpleModel", model=model)
