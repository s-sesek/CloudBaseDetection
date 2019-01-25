import math
import os.path
import sys
import tensorflow as tf
import collections
import six


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder', './trainingdata/images', 'Folder containing images.')

tf.app.flags.DEFINE_string('semantic_segmentation_folder', './trainingdata/maskedimages', 'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string('list_folder', './trainingdata/', 'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string('output_dir', './tfrecords', 'Path to save converted SSTable of TensorFlow examples.')

# Format of the input images
tf.app.flags.DEFINE_enum('image_format', 'png', ['jpg', 'jpeg', 'png'], 'Image format.')

# Format of the pre-segmented images
tf.app.flags.DEFINE_enum('label_format', 'png', ['png'], 'Segmentation label format.')

# A map from image format to expected data format.
_IMAGE_FORMAT_MAP = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
}


_NUM_SHARDS = 4


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, test).
  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = ImageReader('jpeg', channels=3)
  label_reader = ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(FLAGS.output_dir, '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:

      start_idx = shard_id * num_per_shard

      end_idx = min((shard_id + 1) * num_per_shard, num_images)

      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, len(filenames), shard_id))

        sys.stdout.flush()

        # Read the image.
        image_filename = os.path.join(FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)

        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()

        height, width = image_reader.read_image_dims(image_data)

        # Read the semantic segmentation annotation.
        seg_filename = os.path.join( FLAGS.semantic_segmentation_folder, filenames[i] + '.' + FLAGS.label_format)

        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()

        seg_height, seg_width = label_reader.read_image_dims(seg_data)

        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')

        # Convert to tf example.
        example = image_seg_to_tfexample(image_data, filenames[i], height, width, seg_data)

        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, image_format='jpeg', channels=3):
    """Class constructor.
    Args:
      image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
      channels: Image channels.
    """
    with tf.Graph().as_default():
      self._decode_data = tf.placeholder(dtype=tf.string)
      self._image_format = image_format
      self._session = tf.Session()
      if self._image_format in ('jpeg', 'jpg'):
        self._decode = tf.image.decode_jpeg(self._decode_data,
                                            channels=channels)
      elif self._image_format == 'png':
        self._decode = tf.image.decode_png(self._decode_data,
                                           channels=channels)

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
    image = self._session.run(self._decode,
                              feed_dict={self._decode_data: image_data})
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

  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


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

def main(unused_argv):
  dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()
