
import tensorflow as tf

class Model(object):
  def __init__(self, args, name="model"):

  def _build_placeholder(self):
    with tf.name_scope("data"):
      self.encoder_input = tf.placeholder(tf.int32, [None, None], name="encoder_data")
      self.encoder_length = tf.placeholder(tf.int32, [None], name="encoder_length")
      self.decoder_input = tf.placeholder(tf.int32, [None, None], name="decoder_input")
      self.decoder_length = tf.placeholder(tf.int32, [None], name="decoder_length")

  def _build_graph(self):

  def _build_seq2seq(self):
    return tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
      
      )
