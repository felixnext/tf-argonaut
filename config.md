# Configuration

The general configuration for each experiment is defined in a `json` file (see `sample_config.json` as an example).
There are three top level structures: `model`, `data` and `training`.

## Templates

As most of the variables between different experiments should not change (to measure the impact of the ones that do change), templates are an important concept to save time!
You can define a list of template files at the root level of the `json` file (alongside `model`, `data` and `training`), which define the relative path to a template file.
These template files are not required to have all or any of the three top level structure and can themselves relate to other templates.
(The path is relative to the current template file).

```json
"templates": [
    "../experiment_tmp/tmp_multitask.json"
]
```

> Keep in mind, that if you define multiple template files they are loaded in the given order and **later ones will overwrite settings of previous loaded templates** (in case they are defined in both files).

## model

The model task defines the general topology used for the task. it contains the following keywords:

* `class` - Path to the model class from the `hormones.models` namespace.
* `params` - List of parameters that is passed as args to the constructor for the model class

## data

Defines datasets and operations applied to these datasets. Each item has an `id` by which it is referred from other datasets or tasks.
There are two types of datasets. Either directly loaded or inherited.

Directly loaded datasets contain the following keywords:

* `dataset` - Path to the dataset loading function from the `hormones.datasets` namespace.
* `params` - Optional: Input arguments for the loading function
* `batch_size` - Optional: If the provided function loads a tfdata dataset, this defines the batch_size for the data

Inherited datasets contain only the `input` keyword, which references the `id` of the input dataset

All datasets can also contain a list of operations, which perform transformations of the input data (such as label changes or augmentations).
These items contain two keywords:

* `name` - name of the operation to use from the `hormones.datasets.utils` namespace
* `params` - Dictionary of parameters that are passed as arguments to the function

## training

The training data contains a list of tasks that are trained sequentially in the given order.
Each task should contain the following keywords:

* `id` - Name of the task
* `epochs` - Number of epochs to train at max
* `batch_size` - Size of the batches for the given dataset
* `shuffle` - Size of the shuffle buffer for tf-data input (Shuffles only training set) - Used esp. for DWA to avoid randomness in anchor calc
* `dataset` - `id` of the dataset that should be used
* `optimizer` - Optimizer that should be used. Contains the `class` as reference name in the `tensorflow.keras.optimizers` namespace and `params` that are passed to this class
* `head` - reference name of the class in the `hormones.components.heads` namespace
* `loss` - Name of the loss to use
* `lr` - Optional Parameters for changing the learning rate (note: this overwrites the learning-rate set in the creation of the optimizer)
    * `type` - Relevant type of the learning rate adjustment (options: [`plateau`, `cyclic`, ])
    * `params` - Parameters regarding ot the type
* `curriculum` - (Optional) Used for curriculum learning callback, values here are passed to the init function of `callbacks.CurriculumCallback`
* `stop_early` - Optional Callback for early stopping. Contains the following keywords:
    * `active` - defines if early stopping should be used (`true` / `false`)
    * `item` - name of the metric to use for early stopping
    * `min_delta` - minimal change to be used for the early stopping to occur
    * `patience` - number of epochs below `min_delta` before stopping training
* `tensorboard` - Optional Callback for tensorboard. Contains the following keywords:
    * `enable` - Bool whether tensorboard should be used
    * `histogram` - Plot weight histograms after each epoch
    * `summaries` - Plot scalar summaries
* `metrics` - List of metric names to use