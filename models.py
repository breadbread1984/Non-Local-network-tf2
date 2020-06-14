#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def NonLocalBlock(hidden_dim = 1024, pool_size = (2,2,2)):

  inputs = tf.keras.Input((None, None, None, hidden_dim)); # inputs.shape = (batch, seq_length, height, width, hidden_dim)
  query = tf.keras.layers.Conv3D(filters = 512, kernel_size = (1,1,1), padding = 'same')(inputs); # query.shape = (batch, seq_length, height, width, 512)
  key = tf.keras.layers.Conv3D(filters = 512, kernel_size = (1,1,1), padding = 'same')(inputs); # key.shape = (batch, seq_length, height, width, 512)
  key = tf.keras.layers.MaxPool3D(pool_size = pool_size)(key); # key.shape = (batch, seq_length, height, width, 512)
  value = tf.keras.layers.Conv3D(filters = 512, kernel_size = (1,1,1), padding = 'same')(inputs); # value.shape = (batch, seq_length, height, width, 512)
  value = tf.keras.layers.MaxPool3D(pool_size = pool_size)(value); # value.shape = (batch, seq_length, height, width, 512)
  logits = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(tf.keras.layers.Reshape((-1, 512))(x[0]), tf.keras.layers.Reshape((-1, 512))(x[1]), 
                                                             transpose_b = True))([query, key]); # logits.shape = (batch, seq_length * height * width, seq_length * height * width)
  attention = tf.keras.layers.Softmax(axis = -1)(logits); # attention.shape = (batch, seq_length * height * width, seq_length * height * width)
  context = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], tf.keras.layers.Reshape((-1, 512))(x[1])))([attention, value]); # context.shape = (batch, seq_length * height * width, 512)
  context = tf.keras.layers.Lambda(lambda x: tf.keras.layers.Reshape((tf.shape(x[1])[1], tf.shape(x[1])[2], tf.shape(x[1])[3], 512))(x[0]))([context, inputs]); # context.shape = (batch, seq_length, height, width, 512)
  results = tf.keras.layers.Conv3D(filters = 1024, kernel_size = (1,1,1), padding = 'same')(context); # results.shape = (batch, seq_length, height, width, 1024);
  results = tf.keras.layers.Add()([inputs, results]); # results.shape = (batch, seq_length, height, width, 1024)
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  nonlocalblock = NonLocalBlock();
  a = tf.constant(np.random.normal(size = (8,10,8,4, 1024)));
  b = nonlocalblock(a);
  nonlocalblock.save('nonlocalblock.h5');
