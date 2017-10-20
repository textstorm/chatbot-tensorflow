
from collections import Counter
import tensorflow as tf
import numpy as np
import sys
import time

#data utils
def load_data(file_dir):
  print_out("Loading data files.")
  start_time =time.time()
  f = open(file_dir, 'r')
  sentences = []
  i = 0
  while i < 100:
    sentence = f.readline()
    if not sentence:
      break

    sentence = sentence.strip().lower()
    sentences.append(sentence)
    i = i+1
  f.close()
  print_out("Loaded %d sentences from files, time %.2fs" % (len(sentences), time.time() - start_time))
  return sentences


def filter_sentences(sentences, whitelist):
  """
    filter out the emoji in a sentence
    whitelist: 
  """
  def filter_sentence(sentence, whitelist):
    return "".join([ch for ch in sentence if ch in whitelist])

  return [filter_sentence(sentence, whitelist) for sentence in sentences] 

def build_vocab(sentences, max_words=None):
  print_out("Buildding vocabulary...")
  word_count = Counter()
  for sentence in sentences:
    for word in sentence.split(" "):
      word_count[word] += 1

  print_out("The dataset has %d different words totally" % len(word_count))
  if not max_words:
    max_words = len(word_count)
  else:
    filter_out_words = len(word_count) - max_words

  word_dict = word_count.most_common(max_words)
  index2word = ["<unk>"] + ["<s>"] + ["</s>"] + [word[0] for word in word_dict]
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])

  print_out("%d words filtered out of the vocabulary and %d words in the vocabulary" % (filter_out_words, max_words))
  return index2word, word2index

def split_data(sentences, limits):
  encoder_input = []
  decoder_input = []
  num_example = len(sentences) // 2
  print_out("The dataset has %d queries and answers tuple" % num_example)

  for i in range(0, len(sentences), 2):
    encoder_len, decoder_len = len(sentences[i].split(" ")), len(sentences[i+1].split(" "))
    if encoder_len >= limits.encoder_min_len and decoder_len >= limits.decoder_min_len:
      if encoder_len <= limits.encoder_max_len and decoder_len <= limits.decoder_max_len:
        encoder_input.append(sentences[i])
        decoder_input.append(sentences[i+1])

  filtered_data_len = len(encoder_input)
  filter_out_len = num_example - filtered_data_len
  print_out('%d tuple filtered out of the raw data' % filter_out_len)
  return encoder_input, decoder_input

def vectorize(encoder_input,  decoder_input, word2index, sort_by_len=False, verbose=True):
  """
    note: the dict is only 50K,words not in dict is 0
    queries: questions after vectorize
    answers: answers after vectorize
    if sort_by_len equal True, documents sorted by length 
  """
  encoder_input_vector = []
  decoder_input_vector = []
  for idx, (enc, dec) in enumerate(zip(encoder_input, decoder_input)):
    encoder_words = enc.split(' ')
    decoder_words = dec.split(' ')
    seq_encoder = [word2index[w] if w in word2index else 0 for w in encoder_words]
    seq_decoder = [word2index[w] if w in word2index else 0 for w in decoder_words]
    encoder_input_vector.append(seq_encoder)
    decoder_input_vector.append(seq_decoder)

    if verbose and (idx % 50000 == 0):
      print_out("Vectorization: processed {}".format(idx))

  def len_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))
  
  if sort_by_len:
    sort_index = len_argsort(vec_queries)
    encoder_input_vector = [encoder_input_vector[i] for i in sort_index]
    decoder_input_vector = [decoder_input_vector[i] for i in sort_index]

  return encoder_input_vector, decoder_input_vector

def de_vectorize(sample_id, index2word):
  """ The reverse process of vectorization"""
  return " ".join([index2word[int(i)] for i in sample_id if i >= 0])

def padding_data(sentences):
  """
    in general,when padding data,first generate all-zero matrix,then for every
    sentence,0 to len(seq) assigned by seq,like pdata[idx, :lengths[idx]] = seq

      pdata: data after zero padding
      lengths: length of sentences
  """
  lengths = [len(s) for s in sentences]
  n_samples = len(sentences)
  max_len = np.max(lengths)
  pdata = np.zeros((n_samples, max_len)).astype('int32')
  for idx, seq in enumerate(sentences):
    pdata[idx, :lengths[idx]] = seq
  return pdata, lengths 

def get_batchidx(n_data, batch_size, shuffle=False):
  """
    batch all data index into a list
  """
  idx_list = np.arange(0, n_data, batch_size)
  if shuffle:
    np.random.shuffle(idx_list)
  bat_index = []
  for idx in idx_list:
    bat_index.append(np.arange(idx, min(idx + batch_size, n_data)))
  return bat_index

def get_batches(queries, answers, batch_size):
  """
    read all data into ram once
  """
  minibatches = get_batchidx(len(queries), batch_size)
  all_bat = []
  for minibatch in minibatches:
    q_bat = [queries[t] for t in minibatch]
    a_bat = [answers[t] for t in minibatch]
    q_pad, q_len = padding_data(q_bat)
    a_pad, a_len = padding_data(a_bat)
    all_bat.append((q_pad, q_len, a_pad, a_len))
  return all_bat

def pad_for_seq2seq(encoder_bat, decoder_bat):
  encoder_lens = [len(s) for s in encoder_bat]
  decoder_lens = [len(s) for s in decoder_bat]
  encoder_max_len = max(encoder_lens)
  decoder_max_len = max(decoder_lens)
  encoder_pad, decoder_pad = [], []

  for enc_line in encoder_bat:
    enc_line += [0] * (max_len - len(enc_line))
    enc_line = list(reversed(enc_line))
    enc_pad = encoder_pad.append(enc_line)

  for dec_line in decoder_bat:
    


def get_batches_for_seq2seq(encoder_input, decoder_input, batch_size):
  minibatches = get_batchidx(len(encoder_input), batch_size)
  all_bat = []
  for minibatch in minibatches:
    encoder_bat = [encoder_input[t] for t in minibatch]
    decoder_bat = [decoder_input[t] for t in minibatch]

def print_time(s, start_time):
  """Take a start time, print elapsed duration, and return a new time."""
  print_out("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
  sys.stdout.flush()
  return time.time()

def print_out(s, f=None, new_line=True):
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print out_s,

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()

#tensorflow utils
def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto
