# -*- coding: utf-8 -*-

from __future__ import absolute_importd
from __future__ import division
from __future__ import print_function

import os
import re
from tensorflow.python.platform import gfile


# 1-1. Define Symbols

# 1-1-1. Define special vocabulary symbols.
_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]
START_VOCAB_dict = dict()
START_VOCAB_dict['with_padding'] = [_PAD, _UNK]
START_VOCAB_dict['no_padding'] = [_UNK]
PAD_ID = 0
UNK_ID_dict = dict()
UNK_ID_dict['with_padding'] = 1
UNK_ID_dict['no_padding'] = 0

# 1-1-2. Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

# 1-2. Define Tokenizers.
# 1-2-1. Basic tokenizer
def basic_tokenizer(sentence):
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment)) 
  return [w for w in words if w] 
# 1-2-2. Naive tokenizer
def naive_tokenizer(sentence):
  return sentence.split()  

# 1-3. Make vocabulary list which are in dictionary
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer = None, normalize_digits = True):
    # vocabulary_path = data_dir + "in_vocab_%d.txt"
    #       data_path = train_path + ".seq.in"
    
    
  if not gfile.Exists(vocabulary_path): # If don`t exist yet
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    
    # 1-3-1. Read data using gfile.
    with gfile.GFile(data_path, mode="r") as f:
      
      # Count the number of line.
      counter = 0
      for line in f:
        counter += 1
        if counter % 1 == 0:
          print("processing line %d" % counter)
        
        # 1-3-2. Tokenize the line.
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        
        # 1-3-3. Normalize digits to zero and Make frequency vocab.
        for w in tokens:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w 
          
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
            
      vocab_list = START_VOCAB_dict['with_padding'] + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
        
      # 1-3-4. Save the vocab in "vocabulary_path"
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n") 



        
# 1-4. Make index dictionary and word list
def initialize_vocab(vocabulary_path):
 
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab] 
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)]) 
    return vocab, rev_vocab 
         # rev_vocab = [word1, word2, ...], vocab = {word1:index1, ...}
                      
  else: # If not exist the file from "3"
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


# 1-5. String Sentence --) Index Sentence
def sentence_to_token_ids(sentence, vocabulary, UNK_ID,
                          tokenizer=None, normalize_digits=True):
 
  # 1-5-1.
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  
  # 1-5-2.
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words] # Use UNK_ID () If doesn`t exist in dic.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words] # Normalize digits

                      
# 1-6. Tokenize data and turn into token_index.
def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True, use_padding=True):
 
  if not gfile.Exists(target_path): # If there exists, Omit
                      
    # 1-6-1.
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocab(vocabulary_path) # vocab = {word1 : idx1 , .. , ..}
    
    # 1-6-2.
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
           
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 3 == 0:
            print("  tokenizing line %d" % counter)
          
          # 1-6-3.
          if use_padding:
            UNK_ID = UNK_ID_dict['with_padding'] # 1
          else:
            UNK_ID = UNK_ID_dict['no_padding'] # 0 --) UNK
                      
          # 1-6-4.
          token_ids = sentence_to_token_ids(line, vocab, UNK_ID, tokenizer,
                                            normalize_digits) # tokenize the data by line
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

                 
# 1-7. Make index vocab (= labelling)
def create_label_vocab(vocabulary_path, data_path):
                      
  if not gfile.Exists(vocabulary_path): # If exist, omit.
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    
    # 1-7-1. Read Data and get list of words.
    with gfile.GFile(data_path, mode="r") as f: 
     
      # line counter                
      counter = 0
      for line in f:
        counter += 1  
        if counter % 3 == 0:
          print("  processing line %d" % counter)
                      
        label = line.strip()
        vocab[label] = 1
      label_list = START_VOCAB_dict['no_padding'] + sorted(vocab) # [UNK, word1, word2, ...]
                      
      # 1-7-2. Labeling (=Indexing)
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for k in label_list:
          vocab_file.write(k + "\n")
                   

# 1-8. string data --) token_index / make word and indec dictionary
def prepare_multi_task_data(data_dir, in_vocab_size, out_vocab_size):
   
    # data_dir = data/ATIS-sample
    # in_vocab_size = 10000 = out_vocab_size
    
    # 1-8-1. Get Path of train/dev/test Data.
    train_path = data_dir + '/train/train'
    dev_path = data_dir + '/valid/valid'
    test_path = data_dir + '/test/test'
    
    # 1-8-2. Get the vocab list and label vocab(=indexing) in dictionary.
    # 1-8-2-1. Make the vocab file path.
    in_vocab_path = os.path.join(data_dir, "in_vocab_%d.txt" % in_vocab_size)
    out_vocab_path = os.path.join(data_dir, "out_vocab_%d.txt" % out_vocab_size)
    label_path = os.path.join(data_dir, "label.txt")
    
    
    # 1-8-2-2. Make vocab list and label vocab(=indexing) in dictionary
    create_vocabulary(in_vocab_path,  train_path + ".seq.in", 
                      in_vocab_size,  tokenizer=naive_tokenizer) # a
    create_vocabulary(out_vocab_path, train_path + ".seq.out", 
                      out_vocab_size, tokenizer=naive_tokenizer) # b      
    create_label_vocab(label_path, train_path + ".label")
    
    # 1-8-3. String data --) Token_index
    # 1-8-3-1. Get the path of indexing train data set.
    in_seq_train_ids_path = train_path + (".ids%d.seq.in" % in_vocab_size) # a
    out_seq_train_ids_path = train_path + (".ids%d.seq.out" % out_vocab_size) # b
    label_train_ids_path = train_path + (".ids.label")

    # 1-8-3-2. String data --) Token_idx
    # for train data
    data_to_token_ids(train_path + ".seq.in", 
                      in_seq_train_ids_path, 
                      in_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(train_path + ".seq.out", 
                      out_seq_train_ids_path, 
                      out_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(train_path + ".label", 
                      label_train_ids_path, 
                      label_path, 
                      normalize_digits=False, 
                      use_padding=False)
    # for dev data
    in_seq_dev_ids_path = dev_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_dev_ids_path = dev_path + (".ids%d.seq.out" % out_vocab_size)
    label_dev_ids_path = dev_path + (".ids.label")

    
    data_to_token_ids(dev_path + ".seq.in", 
                      in_seq_dev_ids_path, 
                      in_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(dev_path + ".seq.out", 
                      out_seq_dev_ids_path, 
                      out_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(dev_path + ".label", 
                      label_dev_ids_path, 
                      label_path, 
                      normalize_digits=False, 
                      use_padding=False)
    # for test data.
    in_seq_test_ids_path = test_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_test_ids_path = test_path + (".ids%d.seq.out" % out_vocab_size)
    label_test_ids_path = test_path + (".ids.label")
    
    data_to_token_ids(test_path + ".seq.in", 
                      in_seq_test_ids_path, 
                      in_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(test_path + ".seq.out", 
                      out_seq_test_ids_path, 
                      out_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(test_path + ".label", 
                      label_test_ids_path, 
                      label_path, 
                      normalize_digits=False, 
                      use_padding=False)
    
    return [(in_seq_train_ids_path,out_seq_train_ids_path,label_train_ids_path),
            (in_seq_dev_ids_path, out_seq_dev_ids_path, label_dev_ids_path),
            (in_seq_test_ids_path, out_seq_test_ids_path, label_test_ids_path),
            (in_vocab_path, out_vocab_path, label_path)]