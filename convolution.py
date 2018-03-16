from __future__ import division

import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_arr = np.zeros([12, 15])
number = 0
for row_idx in range(input_arr.shape[0]):
  for col_idx in range(input_arr.shape[1]):
    input_arr[row_idx][col_idx] = number
    number += 1

number = 6
w_arr = np.zeros((2, 3))
for row_idx in range(w_arr.shape[0]):
  for col_idx in range(w_arr.shape[1]):
    w_arr[row_idx][col_idx] = number
    number -= 1

strides = [1, 1, 1, 1]

res_shape_0 = int(
    math.ceil((input_arr.shape[0] - w_arr.shape[0]+1)/strides[1]))
res_shape_1 = int(
    math.ceil((input_arr.shape[1] - w_arr.shape[1]+1)/strides[2]))
validation_res = np.zeros(shape=(res_shape_0, res_shape_1))

for row_idx in range(validation_res.shape[0]):
  for col_idx in range(validation_res.shape[1]):
    patch = input_arr[row_idx:row_idx +
                      w_arr.shape[0], col_idx:col_idx+w_arr.shape[1]]
    res = np.sum(patch*w_arr)
    validation_res[row_idx][col_idx] = res

print('result of convolution from its definition: validation_res')
print(validation_res)
pd.DataFrame(validation_res).to_csv(
    'Results/validation_res.csv', index=False, header=False)

input_arr = np.reshape(
    input_arr, [1, input_arr.shape[0], input_arr.shape[1], 1])
w_arr = np.reshape(w_arr, [w_arr.shape[0], w_arr.shape[1], 1, 1])

# 输入图片
net_in = tf.constant(input_arr, tf.float32)
# 卷积核
W = tf.constant(w_arr, tf.float32)
# Valid卷积的结果
result_conv_valid = tf.nn.conv2d(net_in, W, strides, 'VALID')
# Same卷积的结果
result_conv_same = tf.nn.conv2d(net_in, W, strides, 'SAME')

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  valid_conv_res = sess.run(result_conv_valid)
  same_conv_res = sess.run(result_conv_same)

valid_conv_res = np.reshape(
    valid_conv_res, [valid_conv_res.shape[1], valid_conv_res.shape[2]])
same_conv_res = np.reshape(
    same_conv_res, [same_conv_res.shape[1], same_conv_res.shape[2]])

pd.DataFrame(valid_conv_res).to_csv(
    'Results/conv_valid.csv', index=False, header=False)
pd.DataFrame(same_conv_res).to_csv(
    'Results/conv_same.csv', index=False, header=False)
