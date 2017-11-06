
#with the .txt data in the same folder

import random
import os

def get_convs(convs_dir=None):
  if not convs_dir:
    convs_dir = "movie_conversations.txt"
  raw_data = open(convs_dir, 'r').readlines()

  convs = []
  for line in raw_data:
    new_line = line.strip('\n').split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    convs.append(new_line.split(","))
  return convs

#attention: if forget "\n", the space in the end will be strip
def get_id2line(line_dir=None):
  if not line_dir:
    line_dir = "movie_lines.txt"
  raw_data = open(line_dir, 'r').readlines()

  id2line = {}
  for line in raw_data:
    new_line = line.strip('\n').split(" +++$+++ ")
    if len(new_line) == 5:
      id2line[new_line[0]] = new_line[4]
  return id2line

def get_dataset(convs, id2line):
  src_utterance = []
  tgr_utterance = []

  for conv in convs:
    if len(conv) % 2 != 0:
      conv = conv[:-1]
    for idx, line_id in enumerate(conv):
      if idx % 2 == 0:
        src_utterance.append(id2line[line_id])
      else:
        tgr_utterance.append(id2line[line_id])

  return src_utterance, tgr_utterance

def save_dataset(src_utterance, tgr_utterance, save_dir="", eval_size=1000, test_size=1000):
  #save files
  train_file = open(os.path.join(save_dir, "train.txt"), 'w')
  eval_file = open(os.path.join(eval_dir, "eval.txt"), 'w')
  test_file = open(os.path.join(save_dir, "text.txt"), 'w')

  eval_test_id = random.sample(list(range(len(src_utterance))), eval_size + test_size)
  eval_id = random.sample(eval_test_id, eval_size)
  test_id = [i for i in eval_test_id if i not in eval_id]

  for i in range(len(src_utterance)):
    if i in test_id:
      test_file.write(src_utterance[i] + '\n')
      test_file.write(tgr_utterance[i] + '\n')
    elif i in eval_id:
      eval_file.write(src_utterance[i], + '\n')
      eval_file.write(tgr_utterance[i], + '\n')
    else
      train_file.write(src_utterance[i] + '\n')
      train_file.write(tgr_utterance[i] + '\n')

  train_file.close()
  eval_file.close()
  test_file.close()

def main():
  convs = get_convs()
  id2line = get_id2line()
  src_utterance, tgr_utterance = get_dataset(convs, id2line)
  save_dataset(src_utterance, tgr_utterance)

if __name__ == '__main__':
  main()