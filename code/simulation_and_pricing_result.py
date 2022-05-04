
from DerivativesProject import *

'''
以
S = 166.960
r = 0.00765
q = 0.027007
T1 = 0.1
T2 = 0.3
K1 = 10
K2 = 100
为例进行蒙特卡洛模拟、最小二乘蒙特卡洛模拟以及三叉树定价
'''
T1 = 0.1
T2 = 0.3
K1 = 10
K2 = 100

print("欧式Y的二叉树定价：")
print(e_Y_price(S,K1,K2,T1,T2,r,q,100,sigma))
print("Y的二叉树定价：")
print(Y_price(S,K1,K2,T1,T2,r,q,100,sigma))
print('Y的三叉树定价：')
print(C_TT_Am_Y(S, K1, K2, T1, T2, r ,q, 100, sigma))
print('蒙特卡洛模拟模拟欧式Y的价格：')
print(Y_price_simulations(S,r,q,T1,T2,sigma,K1,K2,1000))#重复1000次实验
print('最小二乘蒙特卡洛模拟模拟Y的价格：')
p =Y_MC_American_call_price(r,q,sigma,S,T1,T2,100,1000,K1,K2)#1000路径，100步
print(p)
