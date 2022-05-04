
import DerivativesProject
from DerivativesProject import *
import numpy as np
import matplotlib.pyplot as plt

#case 1 讨论K2变化
plt.figure()
T1 = 0.1
T2 = 0.3
K1 = 10
K2 = range(50,251,5)
n =100
eY = [e_Y_price(S,K1,k2,T1,T2,r,q,n,sigma) for k2 in K2]
Y = [Y_price(S,K1,k2,T1,T2,r,q,n,sigma) for k2 in K2]
#d = [(Y_price(S,K1,k2,T1,T2,r,q,n,sigma)-e_Y_price(S,K1,k2,T1,T2,r,q,n,sigma)) for k2 in K2]
print('欧式Y的价格为：')
print(eY)
plt.plot(K2, eY, label='eY')
print('Y的价格为：')
print(Y)
plt.plot(K2, Y, label='Y')
#plt.plot(K2, c, label='c')
plt.xlabel('K2')
plt.ylabel('Y price')
plt.legend(loc=1)

#case 2:讨论T2变化
plt.figure()
T1 = 0.1
T2 = [i/10 for i in range(1,11)]
K1 = 10
K2 = 100
n =100
eY = [e_Y_price(S,K1,K2,T1,t2,r,q,n,sigma) for t2 in T2]
Y = [Y_price(S,K1,K2,T1,t2,r,q,n,sigma) for t2 in T2]
c = [c_BS(S,K2,t2+T1,r,q,sigma)-K1 for t2 in T2]
d = [(Y_price(S,K1,K2,T1,t2,r,q,n,sigma)-e_Y_price(S,K1,K2,T1,t2,r,q,n,sigma)) for t2 in T2]
print('欧式Y的价格为：')
print(eY)
plt.plot(T2, eY, label='eY')
print('Y的价格为：')
print(Y)
plt.plot(T2, Y, label='Y')
#plt.plot(T2, d, label='d')
#plt.plot(T2, c, label='C-K1')
plt.xlabel('T2')
plt.ylabel('Y price')
plt.legend(loc=1)

#case 3 讨论n变化
plt.figure()
T1 = 0.1
T2 = 0.3
K1 = 10
K2 = 120
n =range(1,200)
eY = [e_Y_price(S,K1,K2,T1,T2,r,q,i,sigma) for i in n]
Y = [Y_price(S,K1,K2,T1,T2,r,q,i,sigma) for i in n]
print('欧式Y的价格为：')
print(eY)
plt.plot(n, eY, label='eY')
print('Y的价格为：')
print(Y)
plt.plot(n, Y, label='Y')
plt.xlabel('N')
plt.ylabel('Y price')
plt.legend(loc=4)

#case 4 讨论K1变化
plt.figure()
T1 = 0.1
T2 = 0.3
K1 = range(0,15)
K2 = 100
n =100
eY = [e_Y_price(S,k1,K2,T1,T2,r,q,n,sigma) for k1 in K1]
Y = [Y_price(S,k1,K2,T1,T2,r,q,n,sigma) for k1 in K1]
#c = [c_BS(S,K2,T2+T1,r,q,sigma)-k1 for k1 in K1]
print('欧式Y的价格为：')
print(eY)
plt.plot(K1, eY, label='eY')
print('Y的价格为：')
print(Y)
plt.plot(K1, Y, label='Y')
#plt.plot(K1, c, label='C-K1')
plt.xlabel('K1')
plt.ylabel('Y price')
plt.legend(loc=1)

#case 5 讨论T1变化
plt.figure()
T1 = [i/10 for i in range(1,11,1)]
T2 = 1
K1 = 10
K2 = 100
n =100
eY = [e_Y_price(S,K1,K2,t1,T2,r,q,n,sigma) for t1 in T1]
Y = [Y_price(S,K1,K2,t1,T2,r,q,n,sigma) for t1 in T1]
print('欧式Y的价格为：')
print(eY)
plt.plot(T1, eY, label='eY')
print('Y的价格为：')
print(Y)
plt.plot(T1, Y, label='Y')
plt.xlabel('T1')
plt.ylabel('Y price')
plt.legend(loc=4)
