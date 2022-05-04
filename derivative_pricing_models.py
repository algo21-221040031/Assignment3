
import numpy as np
from scipy.stats import norm
from random import gauss
import math
from math import exp, sqrt,log
import matplotlib.pyplot as plt
import pandas as pd
from __future__ import division
'''
S:当前股价
K:执行价
T:到期时间
r:无风险利率
q:红利率
sigma:波动率
N:二叉树步数
dt:单位时间间隔
'''
#欧式看涨期权的BS定价用于验证二叉树定价的准确性
def c_BS(S,K,T,r,q,sigma):
    sigma_sqrt_t = sigma*(T**0.5)
    df = r-q+0.5*sigma**2
    df1 = (math.log(S/K) + df*T)/sigma_sqrt_t
    df2 = df1 - sigma_sqrt_t
    nd1 = norm.cdf(df1)
    nd2 = norm.cdf(df2)
    call_price = S*math.exp(-q*T)*nd1 -K*math.exp(-r*T)*nd2
    return call_price
#欧式看涨期权的二叉树定价  
def e_bina_call(S,K,T,r,q,N,sigma):
    dt = T/N
    #u = math.exp( sigma*(dt**0.5))
    #d = math.exp(- sigma*(dt**0.5))
    u = math.exp((r-q)*dt + sigma*(dt**0.5))
    d = math.exp((r-q)*dt - sigma*(dt**0.5))
    p = (math.exp((r-q)*dt)-d)/(u-d)
    lat = np.zeros((N+1,N+1))
    for j in range(0,N+1):
        lat[N,j] = max(0,S*(u**j)*(d**(N-j)) - K)
    for i in range(N-1,-1,-1):
        for j in range(0,i+1):
            lat[i,j] = math.exp(-r*dt)*(p* lat[i+1,j+1] + (1-p) * lat[i+1,j])
    return lat[0,0]
#美式看涨期权的二叉树定价
def a_bina_call(S,K,T,r,q,N,sigma):
    dt = T/N
    u = math.exp((r-q)*dt + sigma*(dt**0.5))
    d = math.exp((r-q)*dt - sigma*(dt**0.5))
    #u = math.exp( sigma*(dt**0.5))
    #d = math.exp(- sigma*(dt**0.5))
    p = (math.exp((r-q)*dt)-d)/(u-d)
    lat = np.zeros((N+1,N+1))
    for j in range(0,N+1):
        lat[N,j] = max(0,S*(u**j)*(d**(N-j)) - K)
    for i in range(N-1,-1,-1):
        for j in range(0,i+1):
            lat[i,j] = max(S*(u**j)*(d**(i-j))-K,math.exp(-r*dt)*(p* lat[i+1,j+1] + (1-p) * lat[i+1,j]))
    return lat[0,0]

#Y期权定价
def Y_price(S,K1,K2,T1,T2,r,q,N,sigma):
    dt = T1/N
    u = math.exp((r-q)*dt + sigma*(dt**0.5))
    d = math.exp((r-q)*dt - sigma*(dt**0.5))
    p = (math.exp((r-q)*dt)-d)/(u-d)
    lat = np.zeros((N+1,N+1))
    for j in range(0,N+1):
        lat[N,j] = max(0,c_BS(S*(u**j)*(d**(N-j)),K2,T2,r,q,sigma)-K1)
        #lat[N,j] = max(0,e_bina_call(S*(u**j)*(d**(N-j)),K2,T2,r,q,100,sigma)-K1)
    for i in range(N-1,-1,-1):
        for j in range(0,i+1):
            
            lat[i,j] = max(c_BS(S*(u**j)*(d**(i-j)),K2,T2,r,q,sigma)-K1,\
                           math.exp(-r*dt)*(p* lat[i+1,j+1] + (1-p) * lat[i+1,j]))
            '''
            lat[i,j] = max(e_bina_call(S*(u**j)*(d**(i-j)),K2,T2,r,q,100,sigma)-K1,\
                           math.exp(-r*dt)*(p* lat[i+1,j+1] + (1-p) * lat[i+1,j]))
            '''
    return lat[0,0]

#若Y为欧式期权的定价    
def e_Y_price(S,K1,K2,T1,T2,r,q,N,sigma):
    dt = T1/N
    u = math.exp((r-q)*dt + sigma*(dt**0.5))
    d = math.exp((r-q)*dt - sigma*(dt**0.5))
    p = (math.exp((r-q)*dt)-d)/(u-d)
    lat = np.zeros((N+1,N+1))
    for j in range(0,N+1):
        lat[N,j] = max(0,c_BS(S*(u**j)*(d**(N-j)),K2,T2,r,q,sigma)-K1)
        #lat[N,j] = max(0,e_bina_call(S*(u**j)*(d**(N-j)),K2,T2,r,q,100,sigma)-K1)
    for i in range(N-1,-1,-1):
        for j in range(0,i+1):
            lat[i,j] = math.exp(-r*dt)*(p* lat[i+1,j+1] + (1-p) * lat[i+1,j])
    return lat[0,0]

#关于复合期权的三叉树定价
def C_TT_Am_Y(S, K1, K2, T1, T2, r ,q, N, sigma):
    
    dt = (T1) / N 
    u = math.exp(sigma * math.sqrt(3 * dt))
    d = math.exp(-sigma * math.sqrt(3 * dt))
    #u = 1 + sigma*math.sqrt(3*h) + 3 * math.pow(sigma, 2) * h/2
    #d = 1 - sigma*math.sqrt(3*h) + 3 * math.pow(sigma, 2) * h/2
    
    p_u = 1/6 + (r - math.pow(sigma, 2)/2) * math.sqrt(dt/(12 *math.pow(sigma, 2)))
    p_m = 2/3
    p_d = 1/6 - (r - math.pow(sigma, 2)/2) * math.sqrt(dt/(12 *math.pow(sigma, 2)))
  
    #Calculate the stock price at each node
    stock = np.zeros((2 * N + 1, N + 1))
    for i in range(0, N+1):
        for j in range(0, 2 * i + 1):
            #每一个node的股价：
            if j <= i:
                stock[j,i] = S * math.pow(u, i - j) * math.pow(d, 0)
            else:
                stock[j,i] = S * math.pow(u, 0) * math.pow(d, j - i)
    
    #Calculate the X price at each node
    X = np.zeros((2 * N + 1, N + 1))
    for i in range(0, N+1):
        for j in range (0, 2 * i + 1):
        #使用hi时刻的股价分布情况计算X的期权价格，使用BS
            X[j,i] = c_BS(stock[j,i], K2, T2,r,  q, sigma)
            #c_BS(S,K,T,r,q,sigma):
     
    #Calculate the Y price at each node
    C_Y_Am = np.zeros((2 * N + 1, N + 1))
    for j in range(0, 2 * N + 1):
        C_Y_Am[j,N] = max(X[j,N]-K1, 0)
        
    for i in range(N-1, -1, -1):
        for j in range (0, 2 * i):
            C_Y_Am[j,i] = max(X[j,i]-K1, math.exp(- r * dt) * (
                p_u * C_Y_Am[j,i+1] + p_m * C_Y_Am[j+1,i+1] 
                + p_d * C_Y_Am[j+2,i+1]))
    
    C_Y_Am[0,0] = max(X[0,0]-K1, math.exp(- r * dt) * (
                p_u * C_Y_Am[0,1] + p_m * C_Y_Am[1,1] 
                + p_d * C_Y_Am[2,1]))
    
    return C_Y_Am[0,0]
#蒙特卡洛模拟欧式
def calculate_S_T(S, sigma, q,r, T):
    #模拟几何布朗运动，算出最后的S_T
    return S * exp((r -q- 0.5 * sigma ** 2) * T + sigma * sqrt(T) * gauss(0.0, 1.0))

def option_payoff(S_T, K):
    #定义欧式期权收益
    return max(S_T - K, 0.0)
      
def option_price(S, q,r, T, sigma, K, simulations):
    #模拟欧式模拟期权收益
    payoffs = []
    discout = exp(-r * T)
    for i in range(simulations):
        S_T = calculate_S_T(S, sigma, q,r, T)
        payoffs.append(
            option_payoff(S_T, K)
        )
    price = discout * sum(payoffs) / float(simulations)
    return price
def Y_price_simulations(S,r,q,T1,T2,sigma,K1,K2,simulations):
    #模拟欧式Y期权的收益
    payoffs = []
    discout = exp(-r * T1)
    for i in range(simulations):
        S_T1= calculate_S_T(S, sigma, q,r, T1)
        #price = c_BS(S_T1,K2,T2-T1,r,q,sigma)
        price = option_price(S_T1, q,r, T2, sigma, K2, simulations)
        payoffs.append(
            option_payoff(price, K1)
        )
    y_price = discout * sum(payoffs) / float(simulations)
    return y_price
#最小二乘蒙特卡洛模拟美式Y
def sample_paths(r, q,sigma, S_0, T1,T2 ,M, N,K2):
    dt = T1/M
    '''
    r: 无风险利率
    sigma: 股价波动率
    S_0: 初始股价
    T: 期权期限
    M: 股价变化过程离散化的段数
    N: 路径数
    '''
    data = S_0*np.ones((N,1))
    for i in range(M):
        normal_variables = np.random.normal(0,1,(N,1))
        ratios = np.exp((r-q-0.5*sigma**2)*dt +normal_variables*sigma*dt**0.5)
        data = np.concatenate((data,data[:,-1:]*ratios),axis=1)
    for i in range(N):
        for j in range(M+1):
            data[i,j] = c_BS(data[i,j],K2,T2,r,q,sigma)
    return data
#最小二乘法拟合每个时间节点
def linear_fitting(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    S0,S1,S2,S3,S4 = len(X),sum(X),sum(X*X),sum(X**3),sum(X**4)
    V0,V1,V2 = sum(Y), sum(Y*X), sum(Y*X*X)
    coeff_mat = np.array([[S0,S1,S2],[S1,S2,S3],[S2,S3,S4]])
    target_vec = np.array([V0,V1,V2])
    inv_coeff_mat = np.linalg.inv(coeff_mat)
    fitted_coeff = np.matmul(inv_coeff_mat,target_vec)
    resulted_Ys = fitted_coeff[0]+fitted_coeff[1]*X+fitted_coeff[2]*X*X
    return resulted_Ys
#模拟出Y的价格
def Y_MC_American_call_price(r, q,sigma, S_0, T1,T2 ,M, N,K1,K2):
    data = sample_paths(r, q,sigma, S_0, T1,T2 ,M, N,K2)
    option_prices = np.maximum(data[:,-1]-K1, 0)
    for i in range(M-1,0,-1):
        option_prices *= np.exp(-r*T1/M)
        option_prices = linear_fitting(data[:,i], option_prices)
        option_prices = np.maximum(option_prices,data[:,i]-K1)
    option_prices *= np.exp(-r*T1/M)
    return np.average(option_prices)

#希腊值计算
#delta计算：股价上升0.01，Y价格的变动
def delta(S0,K1,K2,T1,T2,r,q,N,sigma):
    S1 = S0+0.01
    '''欧式'''
    e0 = e_Y_price(S0,K1,K2,T1,T2,r,q,N,sigma)
    e1 = e_Y_price(S1,K1,K2,T1,T2,r,q,N,sigma)
    e_delta = (e1-e0)/0.01
    '''美式'''
    a0 = Y_price(S0,K1,K2,T1,T2,r,q,N,sigma)
    a1 = Y_price(S1,K1,K2,T1,T2,r,q,N,sigma)
    a_delta = (a1-a0)/0.01
    return (e_delta,a_delta)

#gamma计算:
def gamma(S0,K1,K2,T1,T2,r,q,N,sigma):
    S1 = S0+0.01
    '''欧式'''
    e_delta0 = delta(S0,K1,K2,T1,T2,r,q,N,sigma)[0]
    e_delta1 = delta(S1,K1,K2,T1,T2,r,q,N,sigma)[0]
    e_gamma = (e_delta1-e_delta0)/0.01
    '''美式'''
    a_delta0 = delta(S0,K1,K2,T1,T2,r,q,N,sigma)[1]
    a_delta1 = delta(S1,K1,K2,T1,T2,r,q,N,sigma)[1]
    a_gamma = a_delta1/0.01 - (a_delta0/0.01)
    return (e_gamma,a_gamma)

#theta值的计算
def theta(S,K1,K2,T1,T2,r,q,N,sigma):
    n_T= T1 -1/365
    '''欧式'''
    e0 = e_Y_price(S,K1,K2,T1,T2,r,q,N,sigma)
    e1 = e_Y_price(S,K1,K2,n_T,T2,r,q,N,sigma)
    e_theta = (e1-e0)*365
    '''美式'''
    a0 = Y_price(S,K1,K2,T1,T2,r,q,N,sigma)
    a1 = Y_price(S,K1,K2,n_T,T2,r,q,N,sigma)
    a_theta = (a1-a0)*365
    return (e_theta,a_theta)

#隐含波动率的迭代计算
def bsm_imp_vol(S,K,T,r,c,q):
    c_est = 0 # 期权价格估计值
    top = 1  #波动率上限
    floor = 0  #波动率下限
    sigma = ( floor + top )/2 #波动率初始值
    count = 0 # 计数器
    while abs( c - c_est ) > 0.000001:
        c_est = c_BS(S,K,T,r,q,sigma)
        #根据价格判断波动率是被低估还是高估，并对波动率做修正
        count += 1       
        if count > 100:
            sigma = 0
            break
        if c - c_est > 0: #f(x)>0
            floor = sigma
            sigma = ( sigma + top )/2
        else:
            top = sigma
            sigma = ( sigma + floor )/2
    return sigma  

#隐含波动率迭代计算:牛顿法

def Vega(S,K,T,r,q,sigma):
    d1 = (math.log(S / K) + (r - q + 1 / 2 * pow(sigma, 2)) * T) / (sigma * math.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

def IV(Market_Price, S, K, r, T, q, initial_value, iterations, precision):
    for i in range(iterations):
        model_price = c_BS(S, K, r, T, initial_value)
        vega = Vega(S, K, r, T, q, initial_value)
        diff = Market_Price - model_price
        if abs(diff) < precision:
            return initial_value
        initial_value += diff / vega
    return initial_value


S = 166.960
r = 0.00765
q = 0.027007
sigma = 0.3007
#K = 120
#T = 0.3835616438356
#N = 100
#sigma = 0.3007