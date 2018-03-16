import tensorflow as tf
def conv_layer(input, channels_in,channels_out):
  w = tf.Variable(tf.zeros([5,5, channels_in,channels_out]))
  b = tf.Variable(tf.zeros([channels_out]))
  # 卷积函数
  # input：卷积的输入图像[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]]
  # w：卷积核，要求一个Tensor[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
  # strides：每一纬的步长
  # SAME，VALID 决定了不同的卷接方式
  conv = tf.nn.conv2d(input,w,strides = [1,1,1,1],padding='SAME')
  act = tf.nn.relu(conv+b)
  return act

def fc_layer(input, channels_in, channels_out):
  w= tf.Variable(tf.zeros([channels_in,channels_out]))
  b = tf.Variable(tf.zeros([channels_out]))
