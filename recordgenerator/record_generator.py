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

"""
This script contains utility functions and classes to converts datasets to TFRecord file format.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""

import collections
import six
import math
import os
import random
import sys
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Number of shards
_NUM_SHARDS = 4

tf.app.flags.DEFINE_enum('image_format', 'png', ['jpg', 'jpeg', 'png'], 'Image format.')
tf.app.flags.DEFINE_enum('label_format', 'png', ['png'], 'Segmentation label format.')

# A map from image format to expected data format.
_IMAGE_FORMAT_MAP = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', }

# Define training images location and tag
tf.app.flags.DEFINE_string('train_input_folder',  # Tag
                           'trainingImages',      # Actual folder name
                           'Folder containing training images')

# Define validation images location and tag
tf.app.flags.DEFINE_string('test_input_folder',  # Tag
                           'validationImages',   # Actual folder name
                           'Folder containing validation images')

# Define training image annotations location and tag
tf.app.flags.DEFINE_string('train_target_folder',  # Tag
                           'trainingAnnotations',  # Actual folder name
                           'Folder containing annotations for training images')

# Define validation annotations location and tag
tf.app.flags.DEFINE_string('test_target_folder',     # Tag
                           'validationAnnotations',  # Actual folder name
                           'Folder containing annotations for validation')

# Define folder location to save the tensor flow record files to
tf.app.flags.DEFINE_string('output_dir',  # Tag
                           'tfRecords',    # Actual folder name
                           'Path to save converted tfrecord of Tensorflow example')


class ImageReader(object):
    #  Reads in an image
    def __init__(self, image_format='png', channels=3):
        """
        Args:
          image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
          channels: Image channels.
        """
        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session()
            if self._image_format in ('jpeg', 'jpg'):
                self._decode = tf.image.decode_jpeg(self._decode_data, channels=channels)
            elif self._image_format == 'png':
                self._decode = tf.image.decode_png(self._decode_data, channels=channels)

    def read_image_dims(self, image_data):
        """Reads the image dimensions.
        Args:
          image_data: string of image data.
        Returns:
          image_height and image_width.
        """
        image = self.decode_image(image_data)
        return image.shape[:2]

    def decode_image(self, image_data):
        """Decodes the image data string.
        Args:
          image_data: string of image data.
        Returns:
          Decoded image data.
        Raises:
          ValueError: Value of image channels not supported.
        """
        image = self._session.run(self._decode, feed_dict={self._decode_data: image_data})

        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('The image channels not supported.')
        return image


def _int64_list_feature(values):
    """Returns a TF-Feature of int64_list.
    Args:
      values: A scalar or list of values.
    Returns:
      A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      A TF-Feature.
    """
    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def image_seg_to_tfexample(image_data, filename, height, width, seg_data):
    """Converts one image/segmentation pair to tf example.
    Args:
     image_data: string of image data.
     filename: image filename.
     height: image height.
     width: image width.
     seg_data: string of semantic segmentation data.

    Returns:
      tf example of one image/segmentation pair.
    """
    return tf.train.Example(features=tf.train.Features(feature={
       'image/encoded': _bytes_list_feature(image_data),
       'image/filename': _bytes_list_feature(filename),
       'image/format': _bytes_list_feature(
           _IMAGE_FORMAT_MAP[FLAGS.image_format]),
       'image/height': _int64_list_feature(height),
       'image/width': _int64_list_feature(width),
       'image/channels': _int64_list_feature(3),
       'image/segmentation/class/encoded': (
           _bytes_list_feature(seg_data)),
       'image/segmentation/class/format': _bytes_list_feature(
           FLAGS.label_format),
    }))


def _convert_dataset(dataset_split, dataset_dir, dataset_label_dir):
    #  Converts the clouds dataset into tfrecord format.
    #  Args:
    #  dataset_split: Dataset split (e.g., train, val).
    #  dataset_dir: Dir in which the dataset locates.
    #  dataset_label_dir: Dir in which the annotations locates.
    #  Raises:
    #  RuntimeError: If loaded image and label have different shape.
    img_names = tf.gfile.Glob(os.path.join(dataset_dir, '*.png'))  # Change from .jpg
    random.shuffle(img_names)
    seg_names = []

    for f in img_names:
        # get the filename without the extension
        basename = os.path.basename(f).split('.')[0]
        target_name=basename.replace("frame", "mask")
        seg = os.path.join(dataset_label_dir, target_name +'.png')
        seg_names.append(seg)

    num_images = len(img_names)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = ImageReader('png', channels=3)  # Change from jpeg
    label_reader = ImageReader('png', channels=1)

    for shard_id in range(_NUM_SHARDS):
        # Generate the file name
        output_filename = os.path.join(FLAGS.output_dir,
                                       '%s-%05d-of-%05d.tfrecord'
                                       % (dataset_split,
                                          shard_id,
                                          _NUM_SHARDS))

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)

            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, num_images, shard_id))
                sys.stdout.flush()
                image_filename = img_names[i]  # Read the image.
                image_data = tf.gfile.FastGFile(image_filename, 'rb').read()   # Make sure 'rb' or utf8 error
                height, width = image_reader.read_image_dims(image_data)
                seg_filename = seg_names[i]  # Read the semantic segmentation annotation.
                seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()  # Make sure 'rb' or utf8 error
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
                example = image_seg_to_tfexample(image_data, img_names[i], height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
    tf.gfile.MakeDirs(FLAGS.output_dir)
    _convert_dataset('train', FLAGS.train_input_folder, FLAGS.train_target_folder)
    _convert_dataset('validation', FLAGS.test_input_folder, FLAGS.test_target_folder)


if __name__ == '__main__':
    tf.app.run()

