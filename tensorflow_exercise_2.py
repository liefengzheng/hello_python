import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import input_data
# mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# 输入图片x是一个2维的浮点数张量。这里，分配给它的shape为[None, 784]，其中784是一张展平的MNIST图片的维度
x = tf.placeholder("float",shape = [None, 784])
y_ = tf.placeholder("float",shape = [None, 10])

# 一个变量代表着TensorFlow计算图中的一个值，能够在计算过程中使用，甚至进行修改
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 变量需要通过seesion初始化后，才能在session中使用
sess.run(tf.global_variables_initializer())
# 1 回归模型 向量化后的图片x和权重矩阵W相乘，加上偏置b，然后计算每个分类的softmax概率值
y = tf.nn.softmax(tf.matmul(x,w) + b)
# 2 训练用的损失函数 tf.reduce_sum把minibatch里的每张图片的交叉熵值都加起来了
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 用TensorFlow进行训练 最速下降法让交叉熵下降，步长为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 往计算图上添加一个新操作，其中包括计算梯度，计算每个参数的步长变化，并且计算出新的参数值
# 每一步迭代，我们都会加载50个训练样本，然后执行一次train_step，并通过feed_dict将x 和 y_张量占位符用训练数据替代
# for i in range(1000):
#   batch = mnist.train.next_batch(50)
#   train_step.run(feed_dict={x:batch[0], y_:batch[1]})

# # tf.argmax能给出某个tensor对象在某一维上的其数据最大值所在的索引值 tf.equal 来检测我们的预测是否真实标签匹配
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# 权重初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape,stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)

# 卷积使用1步长（stride size），0边距（padding size）的模板
def conv2d(x,w):
  return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

# 池化用简单传统的2x2大小的模板做max pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 第一层卷积 由一个卷积接一个max pooling完成
# 前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减少过拟合，我们在输出层之前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
for i in range(5000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
