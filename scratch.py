
"""Contains TF-Slim code for training models.
This script contains various functions for training models. These include
manipulating gradients, creating a `train_op` (an operation that computes the
loss and applies the gradients) and a training loop function. The training loop
allows the user to pass in the `train_op` and runs the optimization according
to user-specified arguments. Note that the training loop uses the
tf.train.Supervisor and its managed_session in its implementation to ensure the
ability of worker processes to recover from failures.
************************************
* A simple working training script *
************************************
  # Load data and create the model:
  images, labels = LoadData(...)
  predictions = MyModel(images)
  # Define the loss:
  slim.losses.log_loss(predictions, labels)
  total_loss = slim.losses.get_total_loss()
  # Define the optimizer:
  optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)
  # Run training.
  slim.learning.train(train_op, my_log_dir)
*************************
* Creating the train_op *
*************************
In order to train, TF-Slim's train loop needs a train_op: an `Operation` that
(a) computes the loss, (b) applies the gradients to update the weights and
(c) returns the value of the loss. slim.learning.create_train_op creates
such an `Operation`. This function also provides the ability to manipulate
the gradients using a few arguments:
  # Create the train_op and clip the gradient norms:
  train_op = slim.learning.create_train_op(
      total_loss,
      optimizer,
      clip_gradient_norm=4)
  # Create the train_op and scale the gradients by providing a map from variable
  # name (or variable) to a scaling coefficient:
  gradient_multipliers = {
    'conv0/weights': 1.2,
    'fc8/weights': 3.4,
  }
  train_op = slim.learning.create_train_op(
      total_loss,
      optimizer,
      gradient_multipliers=gradient_multipliers)
****************************************************************
* Performing additional (non-gradient) updates during training *
****************************************************************
Many networks utilize modules, like BatchNorm, that require performing a series
of non-gradient updates during training. slim.learning.create_train_op allows
a user to pass in a list of update_ops to call along with the gradient updates.
  train_op = slim.learning.create_train_op(total_loss, optimizer, update_ops)
By default, slim.learning.create_train_op includes all update ops that are
part of the `tf.GraphKeys.UPDATE_OPS` collection. Additionally, TF-Slim's
slim.batch_norm function adds the moving mean and moving variance updates to
this collection. Consequently, users who want to use slim.batch_norm will not
need to take any additional steps in order to have the moving mean and moving
variance updates be computed.
However, users with additional, specialized updates can either override the
default update ops or simply add additional update ops to the
`tf.GraphKeys.UPDATE_OPS` collection:
  # Force TF-Slim NOT to use ANY update_ops:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=[])
  # Use an alternative set of update ops:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=my_other_update_ops)
  # Use an alternative set of update ops in addition to the default updates:
  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, my_update0)
  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, my_update1)
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer)
  # Which is the same as:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
******************************************
* Initializing a model from a checkpoint *
******************************************
It is common to want to 'warm-start' a model from a pre-trained checkpoint.
TF-Slim provides a convenient mechanism for doing so:
  ...
  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)
  # Create the initial assignment op
  checkpoint_path = '/path/to/old_model_checkpoint'
  variables_to_restore = slim.get_model_variables()
  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)
  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)
  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)
***************************************************************************
* Initializing a model from a checkpoint whose variable names don't match *
***************************************************************************
At times, a user may want to initialize a new model with values from a
checkpoint whose variable names do not match those of the current model. In this
case, one needs to create a mapping from the checkpoint variable names to the
current model variables. This requires only a small modification of the code
above:
  ...
  # Creates a model with two variables, var0 and var1
  predictions = MyModel(images)
  ...
  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)
  checkpoint_path = '/path/to/old_model_checkpoint'
  # Create the mapping:
  variables_to_restore = {
      'name_var_0_in_checkpoint': slim.get_unique_variable('var0'),
      'name_var_1_in_checkpoint': slim.get_unique_variable('var1')
  }
  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)
  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)
  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)
*************************************************
* Fine-Tuning Part of a model from a checkpoint *
*************************************************
Rather than initializing all of the weights of a given model, we sometimes
only want to restore some of the weights from a checkpoint. To do this, one
need only filter those variables to initialize as follows:
  ...
  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)
  checkpoint_path = '/path/to/old_model_checkpoint'
  # Specify the variables to restore via a list of inclusion or exclusion
  # patterns:
  variables_to_restore = slim.get_variables_to_restore(
      include=["conv"], exclude=["fc8", "fc9])
  # or
  variables_to_restore = slim.get_variables_to_restore(exclude=["conv"])
  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)
  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)
  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)
******************************************************
* Initializing model variables from values in memory *
******************************************************
One may want to initialize the weights of a model from values from an arbitrary
source (a text document, matlab file, etc). While this is technically feasible
using plain TensorFlow, it also results in the values of your weights being
stored in the graph. For large models, this becomes prohibitively large. TF-Slim
allows you to perform this initial assignment without having to store the values
of the initial model in the graph itself by using placeholders and a feed
dictionary:
  ...
  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)
  # Create the mapping from variable names to values:
  var0_initial_value = ReadFromDisk(...)
  var1_initial_value = ReadFromDisk(...)
  var_names_to_values = {
    'var0': var0_initial_value,
    'var1': var1_initial_value,
  }
  init_assign_op, init_feed_dict = slim.assign_from_values(var_names_to_values)
  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)
  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from tensorflow.contrib.training.python.training import training
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util

__all__ = [
    'add_gradients_summaries', 'clip_gradient_norms', 'multiply_gradients',
    'create_train_op', 'train_step', 'train'
]


def clip_gradient_norms(gradients_to_variables, max_norm):
  """Clips the gradients by the given value.
  Args:
    gradients_to_variables: A list of gradient to variable pairs (tuples).
    max_norm: the maximum norm value.
  Returns:
    A list of clipped gradient to variable pairs.
  """
  clipped_grads_and_vars = []
  for grad, var in gradients_to_variables:
    if grad is not None:
      if isinstance(grad, ops.IndexedSlices):
        tmp = clip_ops.clip_by_norm(grad.values, max_norm)
        grad = ops.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad = clip_ops.clip_by_norm(grad, max_norm)
    clipped_grads_and_vars.append((grad, var))
  return clipped_grads_and_vars


def multiply_gradients(grads_and_vars, gradient_multipliers):
  """Multiply specified gradients.
  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    gradient_multipliers: A map from either `Variables` or `Variable` op names
      to the coefficient by which the associated gradient should be scaled.
  Returns:
    The updated list of gradient to variable pairs.
  Raises:
    ValueError: If `grads_and_vars` is not a list or if `gradient_multipliers`
    is empty or None or if `gradient_multipliers` is not a dictionary.
  """
  if not isinstance(grads_and_vars, list):
    raise ValueError('`grads_and_vars` must be a list.')
  if not gradient_multipliers:
    raise ValueError('`gradient_multipliers` is empty.')
  if not isinstance(gradient_multipliers, dict):
    raise ValueError('`gradient_multipliers` must be a dict.')

  multiplied_grads_and_vars = []
  for grad, var in grads_and_vars:
    if var in gradient_multipliers or var.op.name in gradient_multipliers:
      key = var if var in gradient_multipliers else var.op.name
      if grad is None:
        raise ValueError('Requested multiple of `None` gradient.')

      multiplier = gradient_multipliers[key]
      if not isinstance(multiplier, ops.Tensor):
        multiplier = constant_op.constant(multiplier, dtype=grad.dtype)

      if isinstance(grad, ops.IndexedSlices):
        tmp = grad.values * multiplier
        grad = ops.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad *= multiplier
    multiplied_grads_and_vars.append((grad, var))
  return multiplied_grads_and_vars


def add_gradients_summaries(grads_and_vars):
  """Add summaries to gradients.
  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
  Returns:
    The list of created summaries.
  """
  summaries = []
  for grad, var in grads_and_vars:
    if grad is not None:
      if isinstance(grad, ops.IndexedSlices):
        grad_values = grad.values
      else:
        grad_values = grad
      summaries.append(
          summary.histogram(var.op.name + '/gradient', grad_values))
      summaries.append(
          summary.scalar(var.op.name + '/gradient_norm',
                         clip_ops.global_norm([grad_values])))
    else:
      logging.info('Var %s has no gradient', var.op.name)

  return summaries


_USE_GLOBAL_STEP = 0


def create_train_op(total_loss,
                    optimizer,
                    global_step=_USE_GLOBAL_STEP,
                    update_ops=None,
                    variables_to_train=None,
                    clip_gradient_norm=0,
                    summarize_gradients=False,
                    gate_gradients=tf_optimizer.Optimizer.GATE_OP,
                    aggregation_method=None,
                    colocate_gradients_with_ops=False,
                    gradient_multipliers=None,
                    check_numerics=True):
  """Creates an `Operation` that evaluates the gradients and returns the loss.
  Args:
    total_loss: A `Tensor` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `_USE_GLOBAL_STEP`, then tf.contrib.framework.global_step() is used.
    update_ops: An optional list of updates to execute. If `update_ops` is
      `None`, then the update ops are set to the contents of the
      `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
      it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`,
      a warning will be displayed.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.trainable_variables().
    clip_gradient_norm: If greater than 0 then the gradients would be clipped
      by it.
    summarize_gradients: Whether or not add summaries for each gradient.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    colocate_gradients_with_ops: Whether or not to try colocating the gradients
      with the ops that generated them.
    gradient_multipliers: A dictionary of either `Variables` or `Variable` op
      names to the coefficient by which the associated gradient should be
      scaled.
    check_numerics: Whether or not we apply check_numerics.
  Returns:
    A `Tensor` that when evaluated, computes the gradients and returns the total
      loss value.
  """
  def transform_grads_fn(grads):
    if gradient_multipliers:
      with ops.name_scope('multiply_grads'):
        grads = multiply_gradients(grads, gradient_multipliers)

    # Clip gradients.
    if clip_gradient_norm > 0:
      with ops.name_scope('clip_grads'):
        grads = clip_gradient_norms(grads, clip_gradient_norm)
    return grads

  return training.create_train_op(
      total_loss=total_loss,
      optimizer=optimizer,
      global_step=global_step,
      update_ops=update_ops,
      variables_to_train=variables_to_train,
      transform_grads_fn=transform_grads_fn,
      summarize_gradients=summarize_gradients,
      gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops,
      check_numerics=check_numerics)


def _wait_for_step(sess, global_step, step):
  """Wait till the global step has reached at least 'step'.
  Args:
    sess: A session.
    global_step: A Tensor.
    step: Int.  The global step to reach.
  """
  while True:
    if training_util.global_step(sess, global_step) >= step:
      break
    time.sleep(1.0)


def train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.
  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.
  Returns:
    The total loss and a boolean indicating whether or not to stop training.
  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  trace_run_options = None
  run_metadata = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

  total_loss, np_global_step = sess.run([train_op, global_step],
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
  time_elapsed = time.time() - start_time

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                           'run_metadata-%d' %
                                                           np_global_step)

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                   np_global_step, total_loss, time_elapsed)

  # TODO(nsilberman): figure out why we can't put this into sess.run. The
  # issue right now is that the stop check depends on the global step. The
  # increment of global step often happens via the train op, which used
  # created using optimizer.apply_gradients.
  #
  # Since running `train_op` causes the global step to be incremented, one
  # would expected that using a control dependency would allow the
  # should_stop check to be run in the same session.run call:
  #
  #   with ops.control_dependencies([train_op]):
  #     should_stop_op = ...
  #
  # However, this actually seems not to work on certain platforms.
  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop


_USE_DEFAULT = 0


def train(logdir,
          config):

#  if train_op is None:
#    raise ValueError('train_op cannot be None.')
#
#  if logdir is None:
#    if summary_op != _USE_DEFAULT:
#      raise ValueError('Cannot provide summary_op because logdir=None')
#    if saver is not None:
#      raise ValueError('Cannot provide saver because logdir=None')
#    if trace_every_n_steps is not None:
#      raise ValueError('Cannot provide trace_every_n_steps because '
#                       'logdir=None')
#
#  if isinstance(sync_optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
#    sync_optimizer = [sync_optimizer]
#  if sync_optimizer is not None and startup_delay_steps > 0:
#    raise ValueError(
#        'startup_delay_steps must be zero when sync_optimizer is supplied.')
#
#  if number_of_steps is not None and number_of_steps <= 0:
#    raise ValueError(
#        '`number_of_steps` must be either None or a positive number.')
#
#  graph = graph or ops.get_default_graph()
#  with graph.as_default():
#    if global_step is None:
#      global_step = training_util.get_or_create_global_step()
#    saver = saver or tf_saver.Saver()
#
#    if sync_optimizer is not None:
#      for opt in sync_optimizer:
#        if not isinstance(opt, sync_replicas_optimizer.SyncReplicasOptimizer):
#          raise ValueError(
#              '`sync_optimizer` must be a tf.train.SyncReplicasOptimizer.')
#
#    with ops.name_scope('init_ops'):
#      if init_op == _USE_DEFAULT:
#        init_op = variables.global_variables_initializer()
#
#      if ready_op == _USE_DEFAULT:
#        ready_op = variables.report_uninitialized_variables()
#
#      if local_init_op == _USE_DEFAULT:
#        local_init_op = control_flow_ops.group(
#            variables.local_variables_initializer(),
#            lookup_ops.tables_initializer())
#
#      if sync_optimizer is not None and isinstance(sync_optimizer, list):
#        with ops.control_dependencies([local_init_op] if local_init_op is
#                                      not None else []):
#          if is_chief:
#            local_init_op = control_flow_ops.group(
#                *[opt.chief_init_op for opt in sync_optimizer])
#          else:
#            local_init_op = control_flow_ops.group(
#                *[opt.local_step_init_op for opt in sync_optimizer])
#        ready_for_local_init_op = control_flow_ops.group(
#            *[opt.ready_for_local_init_op for opt in sync_optimizer])
#      else:
#        ready_for_local_init_op = None
#
#    if summary_op == _USE_DEFAULT:
#      summary_op = summary.merge_all()
#
#    if summary_writer == _USE_DEFAULT:
#      summary_writer = supervisor.Supervisor.USE_DEFAULT
#
#    if is_chief and sync_optimizer is not None:
#      # Need to create these BEFORE the supervisor finalizes the graph:
#      init_tokens_op = [opt.get_init_tokens_op() for opt in sync_optimizer]
#      chief_queue_runner = [
#          opt.get_chief_queue_runner() for opt in sync_optimizer]
#
#    if train_step_kwargs == _USE_DEFAULT:
#      with ops.name_scope('train_step'):
#        train_step_kwargs = {}
#
#        if number_of_steps:
#          should_stop_op = math_ops.greater_equal(global_step, number_of_steps)
#        else:
#          should_stop_op = constant_op.constant(False)
#        train_step_kwargs['should_stop'] = should_stop_op
#        if log_every_n_steps > 0:
#          train_step_kwargs['should_log'] = math_ops.equal(
#              math_ops.mod(global_step, log_every_n_steps), 0)
#        if is_chief and trace_every_n_steps is not None:
#          train_step_kwargs['should_trace'] = math_ops.equal(
#              math_ops.mod(global_step, trace_every_n_steps), 0)
#          train_step_kwargs['logdir'] = logdir




    tf.train.MonitoredTrainingSession(
        master='',
        is_chief=is_chief,
        checkpoint_dir=logdir,
        scaffold=None,
        hooks=None,
        chief_only_hooks=None,
        save_checkpoint_secs=10,  #will need to change
        save_summaries_steps=USE_DEFAULT,
        save_summaries_secs=10,  #will need to change
        config=config,
        stop_grace_period_secs=120,
        log_step_count_steps=100,
        max_wait_secs=7200,
        save_checkpoint_steps=USE_DEFAULT
    )

#
#  if summary_writer is not None:
#    train_step_kwargs['summary_writer'] = sv.summary_writer
#
#  total_loss = None
#  should_retry = True
#  while should_retry:
#    try:
#      should_retry = False
#      with sv.managed_session(
#          master, start_standard_services=False, config=session_config) as sess:
#        logging.info('Starting Session.')
#        if session_wrapper is not None:
#          logging.info(
#              'Wrapping session with wrapper function: %s', session_wrapper)
#          sess = session_wrapper(sess)
#        if is_chief:
#          if logdir:
#            sv.start_standard_services(sess)
#        elif startup_delay_steps > 0:
#           # (use sys.maxsize because sys.maxint doesn't exist in Python 3)
#          _wait_for_step(sess, global_step,
#                         min(startup_delay_steps, number_of_steps or
#                             sys.maxsize))
#        threads = sv.start_queue_runners(sess)
#        logging.info('Starting Queues.')
#        if is_chief and sync_optimizer is not None:
#          sv.start_queue_runners(sess, chief_queue_runner)
#          sess.run(init_tokens_op)
#        try:
#          while not sv.should_stop():
#            total_loss, should_stop = train_step_fn(
#                sess, train_op, global_step, train_step_kwargs)
#            if should_stop:
#              logging.info('Stopping Training.')
#              sv.request_stop()
#              break
#        except errors.OutOfRangeError as e:
#          # OutOfRangeError is thrown when epoch limit per
#          # tf.train.limit_epochs is reached.
#          logging.info('Caught OutOfRangeError. Stopping Training. %s', e)
#        if logdir and sv.is_chief:
#          logging.info('Finished training! Saving model to disk.')
#          sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
#          sv.stop(
#              threads,
#              close_summary_writer=True,
#              ignore_live_threads=ignore_live_threads)
#
#    except errors.AbortedError:
#      # Always re-run on AbortedError as it indicates a restart of one of the
#      # distributed tensorflow servers.
#      logging.info('Retrying training!')
#      should_retry = True

  return total_loss

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
tf.app.run()