# Copyright 2018 Edward Allums, Gopal Godhani, Dan Mayich, and Sarah Sesek
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Visualizes the segmentation results via specified color map.

Visualizes the semantic segmentation results by the color map
defined by the different datasets. Supported colormaps are:

* ADE20K (http://groups.csail.mit.edu/vision/datasets/ADE20K/).

* Cityscapes dataset (https://www.cityscapes-dataset.com).

* PASCAL VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/).
"""

import numpy as np

# Dataset names.
_CLOUDS = 'clouds'
_CITYSCAPES = 'cityscapes'
_MAPILLARY_VISTAS = 'mapillary_vistas'
_PASCAL = 'pascal'

# Max number of entries in the colormap for each dataset.
_DATASET_MAX_ENTRIES = {
    _CLOUDS: 4,
    _CITYSCAPES: 19,
    _MAPILLARY_VISTAS: 66,
    _PASCAL: 256,
}


def create_clouds_label_colormap():
  """Creates a label colormap used in ADE20K segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [0,0,0]
      [0, 191, 255],
      [103, 103, 103],
      [231, 231, 231],
  ])


def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [128, 64, 128],
      [244, 35, 232],
      [70, 70, 70],
      [102, 102, 156],
      [190, 153, 153],
      [153, 153, 153],
      [250, 170, 30],
      [220, 220, 0],
      [107, 142, 35],
      [152, 251, 152],
      [70, 130, 180],
      [220, 20, 60],
      [255, 0, 0],
      [0, 0, 142],
      [0, 0, 70],
      [0, 60, 100],
      [0, 80, 100],
      [0, 0, 230],
      [119, 11, 32],
  ])


def create_mapillary_vistas_label_colormap():
  """Creates a label colormap used in Mapillary Vistas segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [165, 42, 42],
      [0, 192, 0],
      [196, 196, 196],
      [190, 153, 153],
      [180, 165, 180],
      [102, 102, 156],
      [102, 102, 156],
      [128, 64, 255],
      [140, 140, 200],
      [170, 170, 170],
      [250, 170, 160],
      [96, 96, 96],
      [230, 150, 140],
      [128, 64, 128],
      [110, 110, 110],
      [244, 35, 232],
      [150, 100, 100],
      [70, 70, 70],
      [150, 120, 90],
      [220, 20, 60],
      [255, 0, 0],
      [255, 0, 0],
      [255, 0, 0],
      [200, 128, 128],
      [255, 255, 255],
      [64, 170, 64],
      [128, 64, 64],
      [70, 130, 180],
      [255, 255, 255],
      [152, 251, 152],
      [107, 142, 35],
      [0, 170, 30],
      [255, 255, 128],
      [250, 0, 30],
      [0, 0, 0],
      [220, 220, 220],
      [170, 170, 170],
      [222, 40, 40],
      [100, 170, 30],
      [40, 40, 40],
      [33, 33, 33],
      [170, 170, 170],
      [0, 0, 142],
      [170, 170, 170],
      [210, 170, 100],
      [153, 153, 153],
      [128, 128, 128],
      [0, 0, 142],
      [250, 170, 30],
      [192, 192, 192],
      [220, 220, 0],
      [180, 165, 180],
      [119, 11, 32],
      [0, 0, 142],
      [0, 60, 100],
      [0, 0, 142],
      [0, 0, 90],
      [0, 0, 230],
      [0, 80, 100],
      [128, 64, 64],
      [0, 0, 110],
      [0, 0, 70],
      [0, 0, 192],
      [32, 32, 32],
      [0, 0, 0],
      [0, 0, 0],
      ])


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((_DATASET_MAX_ENTRIES[_PASCAL], 3), dtype=int)
  ind = np.arange(_DATASET_MAX_ENTRIES[_PASCAL], dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= bit_get(ind, channel) << shift
    ind >>= 3

  return colormap


def get_clouds_name():
  return _CLOUDS


def get_cityscapes_name():
  return _CITYSCAPES


def get_mapillary_vistas_name():
  return _MAPILLARY_VISTAS


def get_pascal_name():
  return _PASCAL


def bit_get(val, idx):
  """Gets the bit value.

  Args:
    val: Input value, int or numpy int array.
    idx: Which bit of the input val.

  Returns:
    The "idx"-th bit of input val.
  """
  return (val >> idx) & 1


def create_label_colormap(dataset=_CLOUDS):
  """Creates a label colormap for the specified dataset.

  Args:
    dataset: The colormap used in the dataset.

  Returns:
    A numpy array of the dataset colormap.

  Raises:
    ValueError: If the dataset is not supported.
  """
  if dataset == _CLOUDS:
    return create_clouds_label_colormap()
  elif dataset == _CITYSCAPES:
    return create_cityscapes_label_colormap()
  elif dataset == _MAPILLARY_VISTAS:
    return create_mapillary_vistas_label_colormap()
  elif dataset == _PASCAL:
    return create_pascal_label_colormap()
  else:
    raise ValueError('Unsupported dataset.')


def label_to_color_image(label, dataset=_PASCAL):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.
    dataset: The colormap used in the dataset.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the dataset color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  if np.max(label) >= _DATASET_MAX_ENTRIES[dataset]:
    raise ValueError('label value too large.')

  colormap = create_label_colormap(dataset)
  return colormap[label]
