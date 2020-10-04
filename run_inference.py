# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os
import csv
import tokenization

import tensorflow.compat.v1 as tf

#from mobilebert import distill_util
#from mobilebert import modeling
#from mobilebert import optimization
import distill_util
import modeling
import optimization 
import numpy as np
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import summary as contrib_summary


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "data_dir", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_predict", False, "Whether to run training.")


flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_bool("use_einsum", True, "Use tf.einsum to speed up.")

flags.DEFINE_bool("use_summary", False, "Use tf.summary to log training.")

flags.DEFINE_string(
    "bert_teacher_config_file", None,
    "The config json file corresponding to the teacher pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "first_input_file", None,
    "First round input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
    "first_max_seq_length", 128,
    "The first round maximum total input sequence length. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("first_train_batch_size", None,
                     "The first round total batch size for training.")

flags.DEFINE_integer("first_num_train_steps", 0,
                     "Number of the first training steps for second round.")

flags.DEFINE_enum("optimizer", "adamw", ["adamw", "lamb"], "The optimizer.")

flags.DEFINE_bool("layer_wise_warmup", False,
                  "Whether to use layer-wise distillation warmup.")

flags.DEFINE_integer("num_distill_steps", 100000,
                     "Number of distillation steps.")

flags.DEFINE_float("distill_temperature", 1.0,
                   "The temperature factor of distill.")

flags.DEFINE_float("distill_ground_truth_ratio", 1.0,
                   "The ground truth factor of distill in 100%.")

flags.DEFINE_float("attention_distill_factor", 0.0,
                   "Whether to use attention distillation.")

flags.DEFINE_float("hidden_distill_factor", 0.0,
                   "Whether to use hidden distillation.")

flags.DEFINE_float("gamma_distill_factor", 0.0,
                   "Whether to use hidden statistics distillation.")

flags.DEFINE_float("beta_distill_factor", 0.0,
                   "Whether to use hidden statistics distillation.")

flags.DEFINE_float("weight_decay_rate", 0.01, "Weight decay rate.")


def _dicts_to_list(list_of_dicts, keys):
  """Transforms a list of dicts to a list of values.

  This is useful to create a list of Tensors to pass as an argument of
  `host_call_fn` and `metrics_fn`, because they take either a list as positional
  arguments or a dict as keyword arguments.

  Args:
    list_of_dicts: (list) a list of dicts. The keys of each dict must include
      all the elements in `keys`.
    keys: (list) a list of keys.

  Returns:
    (list) a list of values ordered by corresponding keys.
  """
  list_of_values = []
  for key in keys:
    list_of_values.extend([d[key] for d in list_of_dicts])
  return list_of_values


def _list_to_dicts(list_of_values, keys):
  """Restores a list of dicts from a list created by `_dicts_to_list`.

  `keys` must be the same as what was used in `_dicts_to_list` to create the
  list. This is used to restore the original dicts inside `host_call_fn` and
  `metrics_fn`.

  Transforms a list to a list of dicts.

  Args:
    list_of_values: (list) a list of values. The length must a multiple of the
      length of keys.
    keys: (list) a list of keys.

  Returns:
    (list) a list of dicts.
  """
  num_dicts = len(list_of_values) // len(keys)
  list_of_dicts = [collections.OrderedDict() for i in range(num_dicts)]
  for i, key in enumerate(keys):
    for j in range(num_dicts):
      list_of_dicts[j][key] = list_of_values[i * num_dicts + j]
  return list_of_dicts

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      use_einsum=False)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  #embedding_table = model.embedding_table
  # logits = tf.matmul(pooled_output, label_embeddings)
  # logits = tf.matmul(pooled_output, label_embeddings)

  return model

def model_fn_builder(bert_config, num_labels, init_checkpoint, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_positions = features['masked_positions']

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, 
        num_labels, use_one_hot_embeddings)
    
    logits, probs = get_logits_output(bert_config, model.get_sequence_output(), model.get_embedding_table(), masked_positions)
    #logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
    values, indices = tf.math.top_k(probs, k=5, sorted=True, name=None)
    #print("logits shape", logits.shape)
    print("input ids", input_ids.shape)
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      if 'predictions' in var.name:
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    output_spec = None
    indices = tf.identity(indices, name="indices")
    probs = tf.identity(values, name="probs")
    output_spec = tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions={"indices": indices, 'probs': probs},
        scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def get_logits(bert_config, input_tensor, output_weights, positions):
  """Get logits for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())

    if bert_config.hidden_size != bert_config.embedding_size:
      extra_output_weights = tf.get_variable(
          name="extra_output_weights",
          shape=[
              bert_config.vocab_size,
              bert_config.hidden_size - bert_config.embedding_size],
          initializer=modeling.create_initializer(
              bert_config.initializer_range))
      output_weights = tf.concat(
          [output_weights, extra_output_weights], axis=1)
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    return logits

def get_logits_output(bert_config, input_tensor, output_weights, positions):
  logits = get_logits(bert_config, input_tensor, output_weights, positions)

  with tf.variable_scope("cls/predictions"):
    log_probs = tf.nn.softmax(logits, axis=-1)


  return logits, log_probs
 
def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         one_hot_labels, true_labels, label_weights):
  """Get loss and log probs for the masked LM."""
  logits = get_logits(bert_config, input_tensor, output_weights, positions)

  with tf.variable_scope("cls/predictions"):
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_weights = tf.reshape(label_weights, [-1])

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

    true_per_example_loss = -tf.reduce_sum(log_probs * true_labels, axis=[-1])
    true_numerator = tf.reduce_sum(label_weights * true_per_example_loss)
    true_denominator = tf.reduce_sum(label_weights) + 1e-5
    true_loss = true_numerator / true_denominator
  return (loss, true_loss, per_example_loss, log_probs, logits)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  # Which indices to gather from the tensor 
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               mask_positions,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.mask_positions = mask_positions
    self.label_id = 0
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=",", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines
  
  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=None))
    return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, max_predictions_per_seq = 20):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        mask_positions=[0] * max_predictions_per_seq,
        label_id=0,
        is_real_example=False)



  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  mask_positions = []
  for position, token in enumerate(tokens):
    if token == '0':
      mask_positions.append(position)
  
  while(len(mask_positions)) < max_predictions_per_seq:
    mask_positions.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(mask_positions) == max_predictions_per_seq

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("mask_positions: %s" % " ".join([str(x) for x in mask_positions]))
  
  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      mask_positions=mask_positions,
      label_id=0,
      is_real_example=True)
  return feature

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features

def input_fn_builder(features, seq_length, is_training, drop_remainder, 
        max_predictions_per_seq = 20):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []
  all_mask_positions = []
  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)
    all_mask_positions.append(feature.mask_positions)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]
    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "masked_positions":
            tf.constant(
                all_mask_positions,
                shape=[num_examples, max_predictions_per_seq],
                dtype=tf.int32)
    })
    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  summary_path = None
  main_summary_path = None
  if FLAGS.use_summary:
    summary_path = os.path.join(FLAGS.output_dir, "my_summary")
    warmup_summary_path = os.path.join(summary_path, "warmup")
    main_summary_path = os.path.join(summary_path, "main")
    tf.gfile.MakeDirs(summary_path)
    tf.gfile.MakeDirs(warmup_summary_path)
    tf.gfile.MakeDirs(main_summary_path)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      keep_checkpoint_max=0,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=2,
      init_checkpoint=FLAGS.init_checkpoint,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)


  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_predict:
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file)
    processor = DataProcessor()
    prediction_examples = processor.get_train_examples(FLAGS.data_dir)[:FLAGS.predict_batch_size]
    input_features = convert_examples_to_features(prediction_examples, [], FLAGS.max_seq_length, tokenizer)
    predict_input_fn = input_fn_builder(features=input_features, seq_length=FLAGS.max_seq_length, is_training=False, drop_remainder=True)
    predictions = estimator.predict(predict_input_fn)
    count = 0
    print(FLAGS.predict_batch_size)
    # for pred in predictions:
    #   count += 1
    # print("number of prediction is ", count)
    stmts = []
    pred  = []

    for example, prediction in zip(prediction_examples, predictions):
      print('%s\t %s\n' % (example.text_a.replace('0', '---'), example.text_b.replace('0', '---')))
      tokens = tokenizer.convert_ids_to_tokens(prediction['indices'])
      probs  = prediction['probs']
      stmts.append(example.text_a.replace('0', '---') +'\t'+example.text_b.replace('0', '---'))
      pred.append(tokens)
      for i in range(len(tokens)):
          print(tokens[i], int(probs[i]*100))
    pred = np.array(pred)
    print(pred.shape)
    #pred = np.reshape(pred, [FLAGS.predict_batch_size, FLAGS.max_predictions_per_seq, 5])       
if __name__ == "__main__":
  flags.mark_flag_as_required("bert_config_file")
  tf.app.run()
