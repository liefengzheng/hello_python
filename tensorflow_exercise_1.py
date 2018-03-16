import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x_data = np.float(1.0)
# y_data = np.dot([0.100,0.200],x_data) + 0.300

# b = tf.Variable(tf.zeros([1],tf.float64))
# w = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
# y = tf.matmul(w,x_data) + b

# loss = tf.reduce_mean(tf.square(y-y_data))

# x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值
x = tf.placeholder("float",[None,784])

# 一个Variable代表一个可修改的张量.可以用于计算输入值，也可以在计算中被修改
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# tf.matmul(​​X，W)表示x乘以W，对应之前等式里面的Wx,这里x是一个2维张量拥有多个输入
y = tf.nn.softmax(tf.matmul(x,w) + b)

y_ = tf.placeholder("float",[None,10])
cross_entorpy = -tf.reduce_sum(y_ * tf.log(y))

# TensorFlow会用你选择的优化算法来不断地修改变量以降低成本,随机梯度下降训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entorpy)

init = tf.global_variables_initializer()
# 在一个Session里面启动模型，并且初始化变量
sess = tf.Session()
sess.run(init)

# 该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step
for i in range(1000):
  batch_xs,batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 计算所学习到的模型在测试数据集上面的正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))