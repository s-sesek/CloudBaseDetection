#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:09:39 2018

@author: sarah
"""

import math
import os
from math import ceil

import cv2
import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from tensorflow.python.framework import ops
#from tensorflow.python.ops import gen_nn_ops
#import io

class DataSet:
    def __init__(self, batch_size='26', folder='data128_128'):
        self.a = 0
        self.batch_size = batch_size
        self.folder = folder

        train_files, test_files = self.train_valid_test_split(
            os.listdir(os.path.join(folder, 'inputs')))

        self.train_inputs, self.train_targets = self.file_paths_to_images(folder, train_files)
        self.test_inputs, self.test_targets = self.file_paths_to_images(folder, test_files, True)

    def file_paths_to_images(self, folder, files_list, verbose=False):
        inputs = []
        targets = []

        for file in files_list:
            input_image = os.path.join(folder, 'inputs', file)
            target_image = os.path.join(folder, 'targets', file)

            test_image = np.array(cv2.imread(input_image))#, 0))  # load grayscale
            # test_image = np.multiply(test_image, 1.0 / 255)
            inputs.append(test_image)

            target_image = cv2.imread(target_image)#, 0)
            #target_image = cv2.threshold(target_image, 127, 1, cv2.THRESH_BINARY)[1]
            targets.append(target_image)

        return inputs, targets

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.7, 0.3)

        N = len(X)
        return (X[:int(ceil(N * ratio[0]))], X[int(ceil(N * ratio[0])): ])

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        inputs = []
        targets = []
        print(self.batch_size, self.pointer, self.train_inputs.shape, self.train_targets.shape)
        for i in range(self.batch_size):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)

    @property
    def test_set(self):
        return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)


    def draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network, batch_num):
        n_examples_to_plot = 12
        fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10))
        fig.suptitle("Accuracy: {}, {}".format(test_accuracy, network.description), fontsize=20)
        for example_i in range(n_examples_to_plot):
            axs[0][example_i].imshow(test_inputs[example_i], cmap='gray')
            axs[1][example_i].imshow(test_targets[example_i].astype(np.float32), cmap='gray')
            axs[2][example_i].imshow(
                np.reshape(test_segmentation[example_i], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]),
                cmap='gray')
    
            test_image_thresholded = np.array(
                [0 if x < 0.5 else 255 for x in test_segmentation[example_i].flatten()])
            axs[3][example_i].imshow(
                np.reshape(test_image_thresholded, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]),
                cmap='gray')
    
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
    
        IMAGE_PLOT_DIR = 'image_plots/'
        if not os.path.exists(IMAGE_PLOT_DIR):
            os.makedirs(IMAGE_PLOT_DIR)
    
        plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
        return buf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
a = tf.constant([10, 20])

with tf.Session() as sess:
  sess.run(a)
def test():
    dataset1=Dataset(batch_size='18', folder = '/data/scripts/LES_TSI/SmallData/')
    print(dataset1.folder)
    print(dataset1.batch_size)
      
if __name__ == '__main__':
    test()#folder='/data/scripts/LES_TSI/SmallData/', batch_size=BATCH_SIZE)
    main()
  self.file_paths_to_images('/data/scripts/LES_TSI/SmallData/inputs/', train_files)

def load_images(image_paths):
    # Load the images from disk.
    images = [imread(path) for path in image_paths]
    # Convert to a numpy array and return it.
   return np.asarray(images)

##########################################################################################################

def train():
    BATCH_SIZE = 40

    network = Network()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # create directory for saving models
    os.makedirs(os.path.join('save', network.description, timestamp))

    dataset = Dataset(folder='data{}_{}'.format(network.IMAGE_HEIGHT, network.IMAGE_WIDTH),
                      batch_size=BATCH_SIZE)

    inputs, targets = dataset.next_batch()
    print(inputs.shape, targets.shape)

#    augmentation_seq = iaa.Sequential([
#        #iaa.Crop(px=(0, 16), name="Cropper"),  # crop images from each side by 0 to 16px (randomly chosen)
#        iaa.Fliplr(0.5, name="Flipper"),
#        iaa.Dropout(0.02, name="Dropout"),
#        iaa.Affine(translate_px={"x": (-network.IMAGE_HEIGHT // 3, network.IMAGE_WIDTH // 3)}, name="Affine")
#    ])

    # change the activated augmenters for binary masks,
    # we only want to execute horizontal crop, flip and affine transformation
#    def activator_binmasks(images, augmenter, parents, default):
#        if augmenter.name in ["Dropout"]:
#            return False
#        else:
#            # default value for all other augmenters
#            return default
#
#    hooks_binmasks = imgaug.HooksImages(activator=activator_binmasks)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('{}/{}-{}'.format('logs', network.description, timestamp),
                                               graph=tf.get_default_graph())
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

        test_accuracies = []
        # Fit all training data
        n_epochs = 5
        global_start = time.time()
        for epoch_i in range(n_epochs):
            dataset.reset_batch_pointer()

            for batch_i in range(dataset.num_batches_in_epoch()):
                batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1

 #               augmentation_seq_deterministic = augmentation_seq.to_deterministic()

                start = time.time()
                batch_inputs, batch_targets = dataset.next_batch()
                batch_inputs = np.reshape(batch_inputs,
                                          (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                batch_targets = np.reshape(batch_targets,
                                           (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))

                batch_inputs = augmentation_seq_deterministic.augment_images(batch_inputs)
                batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

                batch_targets = augmentation_seq_deterministic.augment_images(batch_targets, hooks=hooks_binmasks)

                cost, _ = sess.run([network.cost, network.train_op],
                                   feed_dict={network.inputs: batch_inputs, network.targets: batch_targets,
                                              network.is_training: True})
                end = time.time()
                print('{}/{}, epoch: {}, cost: {}, batch time: {}'.format(batch_num,
                                                                          n_epochs * dataset.num_batches_in_epoch(),
                                                                          epoch_i, cost, end - start))

                if batch_num % BATCH_SIZE == 0 or batch_num == n_epochs * dataset.num_batches_in_epoch():
                    test_inputs, test_targets = dataset.test_set
                    # test_inputs, test_targets = test_inputs[:100], test_targets[:100]

                    test_inputs = np.reshape(test_inputs, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                    test_targets = np.reshape(test_targets, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                    test_inputs = np.multiply(test_inputs, 1.0 / 255)

                    print(test_inputs.shape)
                    summary, test_accuracy = sess.run([network.summaries, network.accuracy],
                                                      feed_dict={network.inputs: test_inputs,
                                                                 network.targets: test_targets,
                                                                 network.is_training: False})

                    summary_writer.add_summary(summary, batch_num)

                    print('Step {}, test accuracy: {}'.format(batch_num, test_accuracy))
                    test_accuracies.append((test_accuracy, batch_num))
                    print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                    max_acc = max(test_accuracies)
                    print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                    print("Total time: {}".format(time.time() - global_start))

                    # Plot example reconstructions
                    n_examples = 12
                    test_inputs, test_targets = dataset.test_inputs[:n_examples], dataset.test_targets[:n_examples]
                    test_inputs = np.multiply(test_inputs, 1.0 / 255)

                    test_segmentation = sess.run(network.segmentation_result, feed_dict={
                        network.inputs: np.reshape(test_inputs,
                                                   [n_examples, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1])})

                    # Prepare the plot
                    test_plot_buf = draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network,
                                                 batch_num)

                    # Convert PNG buffer to TF image
                    image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)

                    # Add the batch dimension
                    image = tf.expand_dims(image, 0)

                    # Add image summary
                    image_summary_op = tf.summary.image("plot", image)

                    image_summary = sess.run(image_summary_op)
                    summary_writer.add_summary(image_summary)

                    if test_accuracy >= max_acc[0]:
                        checkpoint_path = os.path.join('save', network.description, timestamp, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=batch_num)


if __name__ == '__main__':
train()
#######################################################################################
def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, test).
  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)

  image_reader = build_data.ImageReader('jpeg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  
  output_filename = os.path.join(FLAGS.output_dir, 'smallData.tfrecord')
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = 0
      end_idx = num_images
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d' % (i + 1))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(FLAGS.image_folder, filenames[i]+'.'+FLAGS.image_format)
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            FLAGS.semantic_segmentation_folder, filenames[i] + '.' + FLAGS.label_format)
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
tf.app.run()
#
######################################################################################
class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      seg_map: Segmentation map of `image`.
    """
    width, height = image.size
    target_size = image.size
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image.size)]})
    seg_map = batch_seg_map[0]
    return seg_map

#def create_pascal_label_colormap():
#  """Creates a label colormap used in PASCAL VOC segmentation benchmark.
#
#  Returns:
#    A Colormap for visualizing segmentation results.
#  """
#  colormap = np.zeros((256, 3), dtype=int)
#  ind = np.arange(256, dtype=int)
#
#  for shift in reversed(range(8)):
#    for channel in range(3):
#      colormap[:, channel] |= ((ind >> channel) & 1) << shift
#    ind >>= 3
#
#  return colormap
#
#
def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray(['sky', 'cloud', 'base'])
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
