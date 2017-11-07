#coding:utf-8
import numpy as np
'''
a = np.array([[1, 1],
          [1, 1]])
print(a)
b = a * a
print(b)
c = np.dot(a,a)
print(c)

xarr=np.array([1.1,1.2,1.3,1.4,1.5])    #两个数值数组
yarr=np.array([2.1,2.2,2.3,2.4,2.5])
cond=np.array([True,False,True,True,False])    #一个布尔数组
result=np.where(cond,xarr,yarr)    #三元表达式

arr=np.random.randn(5,4)
print(arr)
print(arr.mean());  print(np.mean(arr));  print(arr.sum());
print(arr.mean(axis=1))    #计算该轴上的统计值（0为列，1为行）

arr = np.random.randn(100)
(arr > 0).sum()    #正值的数量
bools = np.array([True,False,True,False])
print(bools.any())    #用于测试数组中是否存在一个或多个True
print(bools.all())    #用于测试数组中所有值是否都是True
'''
arr = np.random.randn(8)
arr.sort()
arr = np.random.randn(5,3)
# arr = np.array(arr)
print(arr)
arr.sort(0)  #二维数组按列排序;  arr.sort(1)  #二维数组按行排序;
print(arr)