import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

si = pd.Series(np.arange(20),index=np.arange(1,21))
print(si)

# data6= pd.DataFrame({'c1':np.random.randn(20),'c2':np.random.randn(20)})
# print(data6)
# data6_cut = pd.cut(data6.c1,4)
# print(pd.value_counts(data6_cut))

# data5=pd.DataFrame({'name':['a','b','c','d','e','f','g'],
#                     'age':[20,30,20,40,50,40,60],
#                     'score':[88,77,66,76,98,100,94],
#                     'sex':['f','m','f','f','m','f','f']})
# print(data5['score'].groupby([data5['sex'],data5['age']]).median())
# print(data5['score'].groupby([data5['sex'],data5['age']]).mean())
# age = [10,14,15,16,19,20,21,22,24,28,32,35,38,41,48,49,50,60,76]
# age_cut = pd.cut(age,[10,20,30,40,50,60])
# print(age_cut.labels)
# print(pd.value_counts(age_cut))
# a = np.zeros((10,10))
# np.concatenate((a,a),axis=0)
# print(np.random.randint(0,100,10))

# data = pd.read_csv(r'pandas_demo\student.csv',nrows=10)
# print(data)
# ta = pd.read_table(r'pandas_demo\student.csv',sep=',')
# print(ta) 

# data1 = pd.DataFrame({'name':['a']*3+['b']*4+['c']*2,
#                       'age':[1,1,2,3,3,4,5,5,5]})
# print(data1.drop_duplicates())
# print(data1.drop_duplicates(['age']))
# print(data1.replace([1,2],np.NaN))
# print(data.dtypes['age'])
# print(data['age'].describe())
# print(data['age'].sum())
# print(data.mean(axis=1))
# data = pd.DataFrame(np.random.randn(10,4),index=np.arange(10),columns=list('ABCD'))
# data = data.cumsum()
# # plt.figure()
# # data.plot.barh
# data.diff().hist(color='k',alpha=0.5,bins=50)
# # data.plot(kind='barh',stacked=True)
# # ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class1')
# # data.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)
# plt.show()
# data = pd.Series(np.random.randn(1000),index = np.arange(1000))
# print(np.random.randn(2,4,3))
# data.cumsum()
# data.plot()
# plt.show()
#print(data.cumsum())
# print(data.cummax())
# print(data.cummin())
# print(data.cumprod())

# left = pd.DataFrame({"Key1":["K0","K1","K2","K3"],
#                      "A":["A0","A1","A2","A3"],
#                      "B":["B0","B1","B2","B3"]})
# right = pd.DataFrame({"Key1":["K0","K1","K2","K4"],
#                      "C":["C0","C1","C2","C3"],
#                      "D":["D0","D1","D2","D3"]})
# print(pd.merge(left,right,how='outer',on=['Key1'],indicator='comment'))
# left = pd.DataFrame(data={'A':['A1','A2','A3'],
#                           'B':['B1','B2','B3']},
#                     index=['K0','K1','K2'])
# right = pd.DataFrame(data={'C':['C1','C2','C3'],
#                           'D':['D1','D2','D3']},
#                     index=['K0','K2','K3'])
# print(pd.merge(left,right,left_index=True,right_index=True,how='outer'))

# boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
# girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
# res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
# print(res)

# data = pd.read_csv(r'pandas_demo\student.csv')
# print(data)
# data.to_pickle(r'pandas_demo\student.pickle')
# dates = pd.date_range('20180315',periods=6)
# df = pd.DataFrame(data = np.arange(24).reshape(6,4),index=dates,columns=['F1','F2','F3','F4'])
# df.iloc[1,1] = np.nan
# df.iloc[2,3] = np.nan
# print(df)

# dfa = df.dropna(axis=0,how='any')
# print(dfa)

# print(df.fillna(value=1000))
# print(df.isna())

#    F1  F2  F3  F4
# 2   0   1   2   3
# 3   4   5   6   7
# 4   8   9  10  11
# 5  12  13  14  15
# 6  16  17  18  19
# 7  20  21  22  23
# s = pd.Series([1,3,6,np.nan,44,1])
# print(s)

# # dates = pd.date_range('20180101',periods=6)
# # df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['f1','f2','f3','f4'])
# # print(df)

# # print(df['f1'])

# # df1 =pd.DataFrame(np.arange(12).reshape(3,4))
# # print(df1)

# df2 = pd.DataFrame({'A':1,'B':pd.Timestamp('20180315'),
#   'C':pd.Series(1,index=list(range(4)),dtype=np.int32),
#   'D':np.array([3]*4,dtype=np.float),
#   'E':pd.Categorical(["first2","second","first1","fourth"]),
#   'F':[15,10,15,20]})

# # print(df2)
# # print(df2.dtypes)
# # print(df2.columns)
# # print(df2.index)
# # print(df2.values)
# # print(df2.describe())
# # print(df2.transpose())

# print(df2.sort_index(axis=0,ascending=False))
# print(df2.sort_values(by=['F','E'],ascending=False))
