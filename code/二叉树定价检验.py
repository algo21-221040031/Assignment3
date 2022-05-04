
from DerivativesProject import e_bina_call,c_BS 
import numpy as np
import matplotlib.pyplot as plt
'''
二叉树定价的准确度检验
使用摩根大通看涨期权数据
2022 4.14 行权时间：0.3835616438356
'''
S = 166.960
r = 0.00765
q = 0.027007
K = 180
T = 0.3835616438356
#N = 100
sigma = 0.3007
n=np.array(range(1,200))
BSM_benchmark =c_BS(S, K, T, r, q,sigma)
b = np.zeros(199)
e = [e_bina_call(S, K, T, r, q,i,sigma) for i in n]
plt.figure(figsize=(9, 5))
plt.plot(n, e, label='CRR')
plt.axhline(BSM_benchmark, color='r', ls='dashed', lw=1.5,label='BSM')
#plt.axhline(BSM_benchmark, color='r', ls='dashed', lw=1.5,label='BSM')
plt.xlabel('steps')
plt.ylabel('europe call premium')
plt.legend(loc=4)
plt.xlim(1, 199)
plt.show()
