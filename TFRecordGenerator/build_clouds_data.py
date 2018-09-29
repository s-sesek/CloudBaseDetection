# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# Converts Clouds data to TFRecord file format with Example protos

import build_data
import math
import os
import random
import sys
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

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

_NUM_SHARDS = 4


def _convert_dataset(dataset_split, dataset_dir, dataset_label_dir):
    #  Converts the clouds dataset into into tfrecord format.
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

    image_reader = build_data.ImageReader('png', channels=3)  # Change from jpeg
    label_reader = build_data.ImageReader('png', channels=1)

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
                example = build_data.image_seg_to_tfexample(image_data, img_names[i], height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
    tf.gfile.MakeDirs(FLAGS.output_dir)
    _convert_dataset('train', FLAGS.train_input_folder, FLAGS.train_target_folder)
    _convert_dataset('validation', FLAGS.test_input_folder, FLAGS.test_target_folder)


if __name__ == '__main__':
    tf.app.run()

