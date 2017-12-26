# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
# linear = tf.contrib.rnn.python.ops.core_rnn_cell_impl
linear = tf.contrib.rnn.BasicRNNCell

# 3-1. 
def attention_single_output_decoder(initial_state, 
                                    attention_states,
                                    output_size=None,
                                    num_heads=1,
                                    dtype=tf.float32,
                                    scope=None,
                                    sequence_length=tf.ones([16]),
                                    initial_state_attention=True,
                                    use_attention=False):

    
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())


  with tf.variable_scope(scope or "decoder_single_output"):
    
    # 3-1-1. attention_states --) hidden
    print (initial_state.eval().shape)
    batch_size = tf.shape(initial_state)[0]             
    print (attention_states.get_shape())
    attn_length = attention_states.get_shape()[1].value # 2 ( = fw, bw)
    attn_size = attention_states.get_shape()[2].value   # 2 * emb_dim
    hidden = tf.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    
    
    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    
    
    # 3-1-2. Using CNN, get weighted hidden features.
    for a in range(num_heads): # xrange --) range
      k = tf.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
      hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(tf.get_variable("AttnV_%d" % a, [attention_vec_size]))

    # 3-1-3. Get attention weight and attention-weighted vector.
    def attention(query, use_attention=False):
        
      """Put attention masks on hidden using hidden_features and query."""
    
      attn_weights = []
      ds = []  # Results of attention reads will be stored here.
      for i in range(num_heads): # xrange --) range
        with tf.variable_scope("Attention_%d" % i):
            
          # 3-1-3-1. Define y
          y = linear(query, attention_vec_size, True)
                          # attention_vec_size = seq_length
          y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
        
          # 3-1-3-2. Define attention weights (=a)
          s = tf.reduce_sum(
              v[i] * tf.tanh(hidden_features[i] + y), [2, 3])
          if use_attention is False: # apply mean pooling
              weights = tf.tile(sequence_length, tf.stack([attn_length]))
              weights = tf.reshape(weights, tf.shape(s))
              a = tf.ones(tf.shape(s), dtype=dtype) / tf.to_float(weights)
          else:
            a = tf.nn.softmax(s)
          attn_weights.append(a)
        
          # 3-1-3-3. Calculate the attention-weighted vector d.
          d = tf.reduce_sum(
              tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
              [1, 2])
          ds.append(tf.reshape(d, [-1, attn_size]))
      return attn_weights, ds

    # 3-1-4. Check?
    batch_attn_size = tf.stack([batch_size, attn_size])
    attns = [tf.zeros(batch_attn_size, dtype=dtype)
             for _ in range(num_heads)] # xrange --) range
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size]) # attn_size = 2 * emb_dim
    
    # 3-1-3-3-1. Get weights and weighted vectors.
    if initial_state_attention:
      attn_weights, attns = attention(initial_state, use_attention=use_attention)
    
    # 3-1-5. Fully Connected Layer
    matrix = tf.get_variable("Out_Matrix", [attn_size, output_size])
                                         # [2*emb_dim, output_size]
    hypothesis = tf.matmul(attns[0], matrix)
    bias_start = 0.0
    bias_term = tf.get_variable("Out_Bias", [output_size],
                                initializer=tf.constant_initializer(bias_start))
    output = hypothesis + bias_term
  return attention_states, attn_weights[0], attns[0], [output] 


# 3-2.
def generate_single_output(encoder_state, attention_states, sequence_length, 
                           targets, num_classes, buckets, 
                           use_mean_attention=False,
                           softmax_loss_function=None, per_example_loss=False, 
                           name=None, use_attention=False):
  all_inputs = targets
  with tf.name_scope(name, "model_with_buckets", all_inputs):
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
      
      # 3-2-1. Get Fully connected output       
      single_outputs = attention_single_output_decoder(encoder_state, 
                                                      attention_states, 
                                                      output_size=num_classes,
                                                      num_heads=1,
                                                   sequence_length=sequence_length,
                                                      use_attention=use_attention)
    
      _, _, _, bucket_outputs = single_outputs
        
                
      # Check!
      if softmax_loss_function is None:
        assert len(bucket_outputs) == len(targets) == 1
        bucket_target = tf.reshape(tf.to_int64(targets[0]), [-1])
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=bucket_outputs[0], labels=bucket_target)
      else:
        assert len(bucket_outputs) == len(targets) == 1
        crossent = softmax_loss_function(bucket_outputs[0], targets[0])
      
    
      # 3-2-2. Get loss using target.
      batch_size = tf.shape(targets[0])[0]
      loss = tf.reduce_sum(crossent) / tf.cast(batch_size, tf.float32)

  return bucket_outputs, loss
