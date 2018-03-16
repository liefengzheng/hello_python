from __future__ import division

import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Mat=np.array([  
        [[1],[4],[-2],[7],[2]],  
        [[3],[8],[1],[0],[3]],  
        [[5],[-6],[-1],[4],[0]],  
        [[7],[-2],[4],[0],[1]],  
        [[2],[6],[1],[3],[-1]]  
    ])  
  
print("The matrix is:",Mat) 

filtKernel=tf.get_variable("weight",[2,2,1,1],initializer=tf.constant_initializer([[1,-2],[0,4]]))  
biases=tf.get_variable("biases",[1],initializer=tf.constant_initializer(2))  
  
Mat=np.asarray(Mat, dtype='float32')  
Mat=Mat.reshape(1,5,5,1)  

x=tf.placeholder('float32',[1,None,None,1])  
conv=tf.nn.conv2d(x,filtKernel,strides=[1,2,2,1],padding="SAME")  
bias=tf.nn.bias_add(conv,biases)  
pools=tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 

with tf.Session() as sess:  
    tf.global_variables_initializer().run()  
    conM=sess.run(conv,feed_dict={x:Mat})  
    re_conM = np.reshape(conM, [3,3])  
    print("converlution matrix: ",re_conM)  
      
    conMbias=sess.run(bias,feed_dict={x:Mat})  
    #print("converlution with bias: ",conMbias)  
      
    pool=sess.run(pools,feed_dict={x:conM})  
    re_pool = np.reshape(pool,[2,2])
    print("the pooling result is: ",re_pool)  



# input_arr = np.zeros([12,15])
# number = 0
# for row_idx in range(input_arr.shape[0]):
#   for col_idx in range(input_arr.shape[1]):
#     input_arr[row_idx][col_idx] = number
#     number +=1
  
# number = 6
# w_arr = np.zeros((2,3))
# for row_idx in range(w_arr.shape[0]):
#   for col_idx in range(w_arr.shape[1]):
#     w_arr[row_idx][col_idx] = number
#     number -=1

# strides = [1,1,1,1]

# res_shape_0 = int(math.ceil((input_arr.shape[0] - w_arr.shape[0]+1)/strides[1]))
# res_shape_1 = int(math.ceil((input_arr.shape[1] - w_arr.shape[1]+1)/strides[2]))
# validation_res = np.zeros(shape = (res_shape_0,res_shape_1))

# for row_idx in range(validation_res.shape[0]):
#   for col_idx in range(validation_res.shape[1]):
#     patch = input_arr[row_idx:row_idx + w_arr.shape[0],col_idx:col_idx+w_arr.shape[1]]
#     res = np.sum(patch*w_arr)
#     validation_res[row_idx][col_idx] = res

# print('result of convolution from its definition: validation_res')
# print(validation_res)
# pd.DataFrame(validation_res).to_csv('Results/validation_res.csv',index=False,header=False)

# input_arr = np.reshape(input_arr,[1,input_arr.shape[0],input_arr.shape[1],1])
# w_arr = np.reshape(w_arr,[w_arr.shape[0],w_arr.shape[1],1,1])

# # 输入图片
# net_in = tf.constant(input_arr,tf.float32)
# # 卷积核
# W = tf.constant(w_arr,tf.float32)
# # Valid卷积的结果
# result_conv_valid = tf.nn.conv2d(net_in,W,strides,'VALID')
# # Same卷积的结果
# result_conv_same = tf.nn.conv2d(net_in,W,strides,'SAME')

# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   valid_conv_res = sess.run(result_conv_valid)
#   same_conv_res = sess.run(result_conv_same)

# valid_conv_res = np.reshape(valid_conv_res,[valid_conv_res.shape[1],valid_conv_res.shape[2]])
# same_conv_res = np.reshape(same_conv_res,[same_conv_res.shape[1],same_conv_res.shape[2]])

# pd.DataFrame(valid_conv_res).to_csv('Results/conv_valid.csv',index=False,header = False)
# pd.DataFrame(same_conv_res).to_csv('Results/conv_same.csv',index=False,header = False)

# a=tf.constant([  
#         [[1.0,2.0,3.0,4.0],  
#         [5.0,6.0,7.0,8.0],  
#         [8.0,7.0,6.0,5.0],  
#         [4.0,3.0,2.0,1.0]],  
#         [[4.0,3.0,2.0,1.0],  
#          [8.0,7.0,6.0,5.0],  
#          [1.0,2.0,3.0,4.0],  
#          [5.0,6.0,7.0,8.0]]  
#     ])

# a = tf.reshape(a,[1,4,4,2])
# pooling = tf.nn.max_pool(a,[1,2,2,1],[1,1,1,1],padding='VALID')
# with tf.Session() as sess:
#   print('image:')
#   image = sess.run(a)
#   print(image)
#   print(image.shape)
  
#   print('result:')
#   result = sess.run(pooling)
#   print(result)
#   print(result.shape)

# input =tf.Variable(tf.random_normal([1,3,3,5]))
# filter = tf.Variable(tf.random_normal([1,1,5,1]))
# op = tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='VALID')
# # print(op)

# input =tf.Variable(tf.random_normal([1,3,3,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,3]))
# op2 = tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='VALID')
# # print(op2)

# input = tf.Variable(tf.random_normal([1,5,5,5]))  
# filter = tf.Variable(tf.random_normal([3,3,5,1]))  
  
# op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')  
# #case 5  
# input = tf.Variable(tf.random_normal([1,5,5,5]))  
# filter = tf.Variable(tf.random_normal([3,3,5,1]))  
  
# op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')  
# #case 6  
# input = tf.Variable(tf.random_normal([1,5,5,5]))  
# filter = tf.Variable(tf.random_normal([3,3,5,7]))  
  
# op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')  

# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#   sess.run(init)
#   print(sess.run(op4))
#   print(sess.run(op5))

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # weights = tf.Variable(tf.random_normal([2,3],stddev=2))
# # print(weights)

# # bias = tf.Variable(tf.zeros([3]))
# # print(bias)

# print(tf.fill([2,3],9))
# h = tf.constant('Hello,Tensorflow!')
# def hello():
#   print('hello liefeng')

# sess = tf.Session()
# print(sess.run(h))

# a = tf.constant(10)
# b = tf.constant(32)
# print(sess.run(a + b))

# sess.close()

# from collections import defaultdict,namedtuple
# dict1 = defaultdict(list)
# # dict1 = dict(list)
# dict1['first'] = 'first'
# dict1['second'] = 'second'

# print(dict1['second'])
# print(dict1['third'])

# student = namedtuple('Student', 'id name gender elder score' )
# students = [student(1, 'Zheng', 'Male', 18, 100),
#   student(2, 'Zhao', 'Female', 18, 90),
#   student(3, 'Qian', 'Female', 18, 90)
# ]

# for person in students:
#   print(student._make(person))
#   print(person)

# import requests

# proxies = {
#   "http": "http://web-proxy.atl.hp.com:8080/",
#   "https": "http://web-proxy.atl.hp.com:8080/",
# }

# r = requests.get('https://www.python.org',proxies = proxies)
# print(r.status_code)
# print(r.content)
# import tornado.httpserver
# import tornado.ioloop
# import tornado.options
# import tornado.web

# from tornado.options import define, options
# define("port", default=8000, help="run on the given port", type=int)


# class IndexHandler(tornado.web.RequestHandler):
#   def get(self):
#     print('receive')
#     greeting = self.get_argument('greeting', 'Hello')
#     para1 = self.get_argument('para1', 'nothing')
#     self.write(greeting + ', welcome you to read: www.itdiffer.com.' + para1)


# class ZhengHandler(tornado.web.RequestHandler):
#   def get(self):
#     name = self.get_argument('name', 'liefeng')
#     self.write(name + ', welcome you to read: localhost')


# if __name__ == "__main__":
#   tornado.options.parse_command_line()
#   handlers = [
#       (r"/", IndexHandler),  # 来自根路径的请求用IndesHandlers处理
#       # 来自/qiwsir/以及其下任何请求（正则表达式表示任何字符）都由QiwsirHandlers处理
#       (r"/zhengliefeng", ZhengHandler),
#   ]
#   app = tornado.web.Application(handlers=handlers)

#   http_server = tornado.httpserver.HTTPServer(app)
#   http_server.listen(options.port)
#   tornado.ioloop.IOLoop.instance().start()
#   print('Sever start')

# import os as os
# print(os.path.abspath('demo.txt'))
# def foo(x,*args,**kargs):
#   print(x)
#   print(args)
#   print(kargs)
#   return {'status':1,'message':2,'concrete':4}

# rtn = foo(1,2,name='zheng')
# print(rtn)

# def newsum(args):
#   if not args:
#     return 0
#   else:
#     return args[0] + newsum(args[1:])

# # print(newsum(range(5)))
# function_lst = [(newsum,range(5)),(newsum,range(10))]
# for funName,para in function_lst:
#   print(funName(para))

# def add(*arg):
#     # print(x)         #输出参数x的值
#     # result = x
#     print(arg)       #输出通过*arg方式得到的值
#     result = 0
#     for i in arg:
#       print(i)
#       result +=i
#       print('result:' + str(result))
#     return result

# print(add(1,2,3,4,5,6,7,8,9))

# print("MY name is:%s\nMy room is:%d\nMy website is:%s"% ('zheng',501,'http://liefeng.zheng.com/private'))
# word = 'python'
# word_list = list(word)
# print(word_list)
# template = 'First={0[0]},Second={0[1]}'
# print(template.format(word_list))

# print(list(map(lambda x,y:x+y,range(9),range(5,12))))

# from functools import reduce

# print(reduce(lambda x,y:x+y, range(10)))
# # print(list(filter(lambda x:x>5, range(10))))
# for gg in filter(lambda x:x>5, range(10)):
#   print(gg)
# coding:utf-8
# num=[1, 2, 3, 4, 5]
# convert_num = [x+2 for x in num]
# str='郑国'
# with open('demo.txt','r') as f:
#   for line in f:
#     print(line,end='')
#   for se in f:
#     print(se,end='\n')
# import os as os
# file_stat = os.stat('demo.txt')
# print(file_stat)
# exec("print('hello world')")
# li = [1,2,3]
# li[1] = 5
# li[3:] = [6]
# la = ['a','b']
# li.extend(la)
# li.append('la')
# print(li)
# if 'la' in li:
#   li.remove('la')

# aliquote = [n for n in range(1,100) if n%3 ==0]
# print(aliquote)

# mybag = [item.strip() for item in [' glass',' apple','green leaf ']]
# print(mybag)

# from calendar import week
# seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# for (i,day) in enumerate(seasons,1):
#   print(i,day)
# def profile():
#     name = "Danny"
#     age = 30
#     return name, age

# chkValue = profile()
# for x in list(chkValue):
#   print(x)
# my_list = [1, 2, 3]
# print(dir(my_list))

# colours = [
#     ('Yasoob', 'Yellow'),
#     ('Ali', 'Blue'),
#     ('Arham', 'Green'),
#     ('Ali', 'Black'),
#     ('Yasoob', 'Red'),
#     ('Ahmed', 'Silver'),
# ]
# # print(dir(colours))
# print(type([]))
# print(type(()))
# print(type({}))

# li = [name for name, value in colours]
# print(li)
# print(set(li))

# my_dict = {
#     'a': 10,
#     'b': 12,
#     'c': 14
# }
# for i in li:
#   print(i)

# kvReverse_dict = {v: k for k, v in my_dict.items()}
# print(kvReverse_dict)

# for n in range(2, 10):
#   for x in range(2, n):
#     if n % x == 0:
#       print(n, 'equals', x, '*', n / x)
#       break
#   else:
#     # loop fell through without finding a factor
#     print(n, 'is a prime number')

# # pprint(kvReverse_dict)
# # from collections import defaultdict

# # favourite_colours = defaultdict(list)

# # for name,colour in colours:
# #   favourite_colours[name].append(colour)
# # print(favourite_colours)
# from collections import defaultdict, Counter, namedtuple

# favs = Counter(name for name,colour in colours)
# print(favs)

# animal = namedtuple('woshizhengliefeng','name age type')
# perry = animal(name = 'perry', age =31, type='cat')
# print(perry.age)
# print(animal.__name__)

# my_list = ['apple', 'banana', 'grapes', 'pear']
# for index, value in enumerate(my_list, 1):
#   print(index, value)
