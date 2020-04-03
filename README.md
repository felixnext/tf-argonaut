# ⛵ tf-argonaut

Library for creating visual experiment pipelines in tensorflow. Allows to test out different network concepts against standardized datasets.

> The Argonauts where a band of heros and adventures that sailed on their ship Argo through the mediterranean and navigated numerous adventures.
> Like one of them this library is designed to turn TF into your own argo and navigate your experiments.

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

TODO: more advanced coding examples

`Argonaut` also comes with various pre-defined models (although you can also easily plug in every keras model, given right input and output structrue).
In particular these models include:

TODO

### Debugging

TODO: integrate options for TF2 debugging

## ⚙ Configuration

Experiments allow you to specify most of the hyper-parameters through a configuration `json` file. See the detailed [configuration guide](config.md) for more details.

## License

This library is provided under the Apache License.

**Pull Requests to improve code quality and add new functionality are more then welcome!**