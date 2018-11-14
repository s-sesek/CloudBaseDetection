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

"""Prepares the data used for DeepLab training/evaluation."""
import tensorflow as tf
from CloudBaseDetection.core import feature_extractor
from CloudBaseDetection.core import preprocess_utils


# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5


def preprocess_image_and_label(image,
                               label,
                               crop_height,
                               crop_width,
                               ignore_label=255,
                               is_training=True,
                               model_variant=None):
  """Preprocesses the image and label.

  Args:
    image: Input image.
    label: Ground truth annotation label.
    crop_height: The height value used to crop the image and label.
    crop_width: The width value used to crop the image and label.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    ignore_label: The label value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.

  Returns:
    original_image: Original image (could be resized).
    processed_image: Preprocessed image.
    label: Preprocessed ground truth segmentation label.

  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and label is None:
    raise ValueError('During training, label must be provided.')
  if model_variant is None:
    tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')

  # Keep reference to original image.
  original_image = image
  processed_image = tf.cast(image, tf.float32)
  if label is not None:
    label = tf.cast(label, tf.int32)
    
  processed_image = tf.reshape(processed_image,[640, 480, 3])
  label= tf.reshape(label, [640, 480, 1])
  # Pad image and label to have dimensions >= [crop_height, crop_width]
  image_shape = tf.shape(processed_image)
  image_height = image_shape[0]
  image_width = image_shape[1]

#  target_height = image_height + tf.maximum(crop_height - image_height, 0)
#  target_width = image_width + tf.maximum(crop_width - image_width, 0)

  # Crop the image and label.
  #if is_training and label is not None: #had to comment out if to run vis.py
  processed_image= preprocess_utils._crop(processed_image, 79, 0, crop_height, crop_width)
  #processed_image = tf.reshape(processed_image,[crop_height, crop_width, 3])
  processed_image.set_shape([crop_height, crop_width, 3])

  if label is not None:
    label= preprocess_utils._crop(label, 79, 0, crop_height, crop_width)
    label= tf.reshape(label, [crop_height, crop_width, 1])

  #reshape image if I pad it - messed up the tensor shape
  # Pad image with mean pixel value.
  #mean_pixel = tf.reshape(
  #    feature_extractor.mean_pixel(model_variant), [1, 1, 3])
#  processed_image = preprocess_utils.pad_to_bounding_box(
#      processed_image, 0, 0, target_height, target_width, [0,0,0])
#
#  if label is not None:
#    label = preprocess_utils.pad_to_bounding_box(
#        label, 0, 0, target_height, target_width, ignore_label)

#
#  if is_training:
#    # Randomly left-right flip the image and label.
#    processed_image, label, _ = preprocess_utils.flip_dim(
#        [processed_image, label], _PROB_OF_FLIP, dim=1)

  return original_image, processed_image, label
