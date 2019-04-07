import math
import six
import tensorflow as tf
from deeplab import common
from deeplab import model
import dataset_info
from deeplab.utils import input_generator

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('eval_logdir', './evallog', 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1, 'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('eval_crop_size', [640, 480], 'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,'How often (in seconds) to run evaluation.')

flags.DEFINE_multi_integer('atrous_rates', [6, 12, 18],'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,'The ratio of input to output spatial resolution.')


flags.DEFINE_multi_float('eval_scales', [1.0],'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,'Add flipped images for evaluation or not.')

# Dataset settings.

flags.DEFINE_string('dataset', 'clouds', 'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val','Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', './tfrecords', 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Get dataset-dependent information.
  dataset = dataset_info.get_dataset(FLAGS.dataset, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)

  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  with tf.Graph().as_default():
    samples = input_generator.get(
        dataset,
        FLAGS.eval_crop_size,
        FLAGS.eval_batch_size,
        dataset_split=FLAGS.eval_split,
        is_training=False,
        model_variant=FLAGS.model_variant)

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=FLAGS.eval_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(samples[common.IMAGE], model_options,
                                         image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)
    predictions = predictions[common.OUTPUT_TYPE]
    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(samples[common.LABEL], shape=[-1])
    weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = tf.where(
        tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

    predictions_tag = 'miou'
    for eval_scale in FLAGS.eval_scales:
      predictions_tag += '_' + str(eval_scale)
    if FLAGS.add_flipped_images:
      predictions_tag += '_flipped'

    # Define the evaluation metric.
    #metric_map = {}
    #metric_map[predictions_tag] = tf.metrics.mean_iou(
    #    predictions, labels, dataset.num_classes, weights=weights)


    metric_map = {}

    indices = tf.squeeze(tf.where(tf.less_equal(
        labels, dataset.num_classes - 1)), 1)
    labels = tf.cast(tf.gather(labels, indices), tf.int32)
    predictions = tf.gather(predictions, indices)

    metric_map[predictions_tag] = tf.metrics.mean_iou(
        predictions, labels, dataset.num_classes, weights=weights)

    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))

    for metric_name, metric_value in six.iteritems(metrics_to_values):
      slim.summaries.add_scalar_summary(
          metric_value, metric_name, print_summary=True)

    num_batches = int(
        math.ceil(dataset.num_samples / float(FLAGS.eval_batch_size)))

    tf.logging.info('Eval num images %d', dataset.num_samples)
    tf.logging.info('Eval batch size %d and num batch %d',
                    FLAGS.eval_batch_size, num_batches)

    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_logdir,
        num_evals=num_batches,
        eval_op=list(metrics_to_updates.values()),
        max_number_of_evaluations=num_eval_iters,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
    tf.app.run()
