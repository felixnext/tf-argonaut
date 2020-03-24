'''
Simple Script to visualize Datasets and its permutations.

author: Felix Geilert
'''

import cv2
import os
from argonaut.datasets import mnist
from argonaut.datasets import utils
from argonaut.datasets import Pipeline

def visualize_dataset(dataset="mnist.mnist_generator", permute=False, show_val=True, size=100):
  # load the relevant data
  # FEAT: Update the script to load any training config and display resulting images / to load any generator and dispaly relevant data
  data_args = [
    {
      "id": "dataset",
      "dataset": dataset,
      "params": {
        "one_hot": False,
        "channels": 3
      },
      "operations": []
    },
    {
      "id": "dataset_tf",
      "input": "dataset",
      "operations": [
        {
          "name": "ExcludeTransformer",
          "params": {
            "count": 2
          }
        },
        {
          "name": "OneHotTransformer",
          "params": {}
        },
        {
          "name": "ResizeTransformer",
          "params": {
            "shape": [size, size]
          }
        },
        {
          "name": "EntropyComplexityTransformer",
          "params": { "wnd_size": 5 }
        },
        {
          "name": "TFDataTransformer",
          "params": {}
        }
      ]
    }
  ]
  pipeline = Pipeline(data_args, [{"id": "test", "dataset": "dataset_tf"}], clean=False)
  data, sum = pipeline.transform()
  print("NEXT")
  # data, sum = mnist.mnist(one_hot=False)

  # # check permutations
  # if permute == True:
  #   data, sum = utils.permute_list(data, 2, summary=sum)
  # else:
  #   data = [data]

  print(sum)
  sum.generate_summary(".")

  for key in data:
    d = data[key]
    if d.type != "tfdata":
      continue
    print("DATASET {}".format(key))
    print(d.num_classes)
    rs = d.unique_classes
    print("Unique Classes:")
    print(rs)
    # iterate through data
    for (x, c), y in d.train:
      cv2.imshow("Mnist", x.numpy())
      print("Class: {}".format(y.numpy()))
      print("Complexity: {}".format(c))
      if cv2.waitKey(0) == 27:
        break

    if show_val == True:
      for (x, c), y in d.test:
        cv2.imshow("Mnist", x.numpy())
        print("Class: {}".format(y.numpy()))
        if cv2.waitKey(0) == 27:
          break

  # clear data
  cv2.destroyAllWindows()
