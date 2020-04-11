# ⛵ tf-argonaut

Library for creating visual experiment pipelines in tensorflow. Allows to test out different network concepts against standardized datasets.

> The Argonauts were a band of heros and adventures that sailed on their ship Argo through the mediterranean and navigated numerous adventures.
> Like one of them this library is designed to turn TF into your own argo and navigate your experiments.
> So set sail!

`Argonaut` was originally build to allow easy research experimentation of multi-task settings against common datasets (thereby reducing the overhead required for experimentation).

## 👶 Getting Started

To install the library simply use PyPi:
```
pip3 install tf-argonaut
```

Alternatively you can install the library directly through `setup.py`:
```
pip3 install .
```

You can then import the library:
```python
import argonaut as argo
```

At its core, `argonaut` allows you to run experiments with a single line of code and a configuration file (see in folder `examples/simple_multitask`):

```python
argo.run_experiment("Baseline", "experiment.json", name="SimpleExample")
```

## 📜 Concepts

Concepts include:

* Experiments
* Pipeline
* Datasets
* Callbacks

### Tools

The library also contains multiple tools that allow to inspect data and quickly start training processes.

## 💾‍ Coding Examples

**Code Examples:** This repository contains several examples on how to use the library and leverage the different concepts in the [example folder](examples/readme.md).

The main idea behind `argonaut` is to reduce the lines of code required to write and test new research ideas.
Therefore most coding samples contain of regular model or layer definitions inside keras/TensorFlow, a configuration file that specifies an experiment and a `run.py` file, which simply contains one line of code to start the experiment.

`Argonaut` also comes with various pre-defined models (although you can also easily plug in every keras model, given right input and output structrue).
In particular these models include:

* `simple.ConvNet` - Simple convolutional network with adjustable number of layers
* `simple.MLP` - Multi-Layer Perceptron with adjustable number of layers
* `alexnet.AlexNet` - The original Alexnet network, as described in [this paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
* `vgg16.VGG16` - The VGG16 network, as described in [this paper](https://arxiv.org/abs/1409.1556)
* `densenet.DenseNet121` - (Other Variants: `169`, `201`, `264`) The DenseNet Variant as described in [this paper](https://arxiv.org/abs/1608.06993)
* `resnetv2.ResNet_v2_50` - (Other Variants: `101`, `152`) The Residual Network Version 2, as described in [this paper](https://arxiv.org/abs/1603.05027)

### Debugging

**Coming Soon:** Options to better integrate the TF2 debugging options are planned (PRs are welcome).

## ⚙ Configuration

Experiments allow you to specify most of the hyper-parameters through a configuration `json` file. See the detailed [configuration guide](config.md) for more details.

## License

This library is provided under the Apache License.

**Pull Requests to improve code quality and add new functionality are more then welcome!**