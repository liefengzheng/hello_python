import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_LOG_LEVEL'] = '2'

input_data = tf.Variable(np.random.rand(10,9,9,3),dtype=tf.float32)
filter_data = tf.Variable(np.random.rand(2,2,3,2),dtype=tf.float32)

y = tf.nn.conv2d(input_data,filter_data,strides=[1,1,1,1],padding='SAME')
print('0. tf.nn.conv2d:',y)

output = tf.nn.avg_pool(value = y,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')
print('1. tf.nn.avg_pool:',output )

output = tf.nn.max_pool(value = y, ksize = [1,2,2,1], strides =[1,1,1,1],padding='SAME')
print('2. tf.nn.max_pool:',output )

output,argmax = tf.nn.max_pool_with_argmax(input=y, ksize = [1,2,2,1], strides =[1,1,1,1],padding='SAME')
print('2.5. tf.nn.max_pool : ', output, argmax)
