import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import numpy as np

# BERT model



class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, score = None):
		self.guid = guid
		self.text_a = text_a
		self.text_b = None
		self.score = score

def create_examples(df, score_column, comment_col_name):
	"""Creates examples for the training and dev sets."""
	examples = []
	for i, row in df.iterrows():
		guid = "comment_%s" % (i)
		text_a = bert.tokenization.convert_to_unicode(row[comment_col_name])
		score = [row[name] for name in score_column]
		examples.append(
			InputExample(guid=guid, text_a=text_a, score = score))
	return examples


def create_tokenizer_from_hub_module(BERT_model_hub):
	"""Get the vocab file and casing info from the Hub module."""
	with tf.Graph().as_default():
		bert_module = hub.Module(BERT_model_hub)
		tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
		with tf.Session() as sess:
			vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],tokenization_info["do_lower_case"]])

	return bert.tokenization.FullTokenizer(
			vocab_file=vocab_file, do_lower_case=do_lower_case)

class PaddingInputExample(object):
	"""Fake example so the num input examples is a multiple of the batch size.
	When running eval/predict on the TPU, we need to pad the number of examples
	to be a multiple of the batch size, because the TPU requires a fixed batch
	size. The alternative is to drop the last batch, which is bad because it means
	the entire output data won't be generated.
	We use this class instead of `None` because treating `None` as padding
	battches could cause silent errors.
	"""

class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, score, is_real_example=True):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.is_real_example = is_real_example
		self.score = score


def convert_examples_to_features(examples, max_seq_length, tokenizer, print_flag = False):
	"""Convert a set of `InputExample`s to a list of `InputFeatures`."""

	features = []
	for (ex_index, example) in enumerate(examples):

		if ex_index % 10000 == 0 and print_flag == True:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		feature = convert_single_example(ex_index, example, max_seq_length, tokenizer, print_flag)
		features.append(feature)
	return features


def convert_single_example(ex_index, example, max_seq_length,
						   tokenizer, print_flag):
	"""Converts a single `InputExample` into a single `InputFeatures`."""

	if isinstance(example, PaddingInputExample):
		return InputFeatures(
			input_ids=[0] * max_seq_length,
			input_mask=[0] * max_seq_length,
			segment_ids=[0] * max_seq_length,
			is_real_example=False)

	tokens_a = tokenizer.tokenize(example.text_a)
	if len(tokens_a) > max_seq_length - 2:
		# print("Num tokens:")
		# print(len(tokens_a
		tokens_a = tokens_a[0:(max_seq_length - 2)]

	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)
	for token in tokens_a:
		tokens.append(token)
		segment_ids.append(0)
	tokens.append("[SEP]")
	segment_ids.append(0)

	input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
	input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)

	assert len(input_ids) == max_seq_length
	assert len(input_mask) == max_seq_length
	assert len(segment_ids) == max_seq_length

	score = example.score

	if ex_index < 5 and print_flag == True:
		tf.logging.info("*** Example ***")
		tf.logging.info("guid: %s" % (example.guid))
		tf.logging.info("tokens: %s" % " ".join(
			[tokenization.printable_text(x) for x in tokens]))
		tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
		tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
		tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
		tf.logging.info("score: %s" % (example.score))

	feature = InputFeatures(
	  input_ids=input_ids,
	  input_mask=input_mask,
	  segment_ids=segment_ids,
	  score = score,
	  is_real_example=True)
	return feature



#again, going to need to create an input_fn_builder that preserves score
def input_fn_builder(features, seq_length, num_labels, is_training, drop_remainder,
					  supplied_context_features=None):
	"""Creates an `input_fn` closure to be passed to TPUEstimator."""

	all_input_ids = []
	all_input_mask = []
	all_segment_ids = []
	all_scores = []

	for feature in features:
		all_input_ids.append(feature.input_ids)
		all_input_mask.append(feature.input_mask)
		all_segment_ids.append(feature.segment_ids)
		all_scores.append(feature.score)

	if supplied_context_features is not None:
		supplied_context_input_ids = []
		supplied_context_input_mask = []
		supplied_context_segment_ids = []
		supplied_context_scores = []

		for feature in supplied_context_features:
			supplied_context_input_ids.append(feature.input_ids)
			supplied_context_input_mask.append(feature.input_mask)
			supplied_context_segment_ids.append(feature.segment_ids)
			supplied_context_scores.append(feature.score)

		supplied_context_input_ids = np.tile(supplied_context_input_ids, (len(features), 1, 1))
		supplied_context_input_mask = np.tile(supplied_context_input_mask, (len(features), 1, 1))
		supplied_context_segment_ids = np.tile(supplied_context_segment_ids, (len(features), 1, 1))
		supplied_context_scores = np.tile(supplied_context_scores, (len(features), 1, 1))

	def input_fn(params):
		"""The actual input function."""
		batch_size = params["batch_size"]

		num_examples = len(features)

		if supplied_context_features is not None:
			num_context_examples = len(supplied_context_features)

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
				"scores":
					tf.constant(all_scores, shape=[num_examples, num_labels],
                        dtype=tf.float32),
				"supplied_context_input_ids":
					tf.constant(
						supplied_context_input_ids, shape=[num_examples, num_context_examples, seq_length],
						dtype=tf.int32),
				"supplied_context_input_mask":
					tf.constant(
						supplied_context_input_mask,
						shape=[num_examples, num_context_examples, seq_length],
						dtype=tf.int32),
				"supplied_context_segment_ids":
					tf.constant(
						supplied_context_segment_ids,
						shape=[num_examples, num_context_examples, seq_length],
						dtype=tf.int32),
				"supplied_context_scores":
					tf.constant(supplied_context_scores, shape=[num_examples, num_context_examples, num_labels],
                        dtype=tf.float32)
			})

		else:
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
				"scores":
					tf.constant(all_scores, shape=[num_examples, num_labels])
			})

		if is_training:
			d = d.repeat()
			# filter
			## choose category at random

			## filter batch dataset to category


			d = d.shuffle(buffer_size=100)

		d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)

		return d

	return input_fn


def find_max_seq_length(features):
	max_len_features = 0
	for feature in features:
	    feature_vec = feature.input_ids
	    len_features = max_seq_length - sum(feature_vec == np.zeros([max_seq_length]))
	    if len_features > max_len_features:
	        max_len_features = len_features
	return max_len_features
