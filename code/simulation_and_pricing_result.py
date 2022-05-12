
from derivative_pricing_models import *

'''
S = 166.960
r = 0.00765
q = 0.027007
T1 = 0.1
T2 = 0.3
K1 = 10
K2 = 100
Monte Carlo Simulation, Binomial Tree Pricing, Trinomial Tree Pricing and Least Squares Monte Carlo Simulation.
'''
T1 = 0.1
T2 = 0.3
K1 = 10
K2 = 100

print("Binomial tree of european compound option:")
print(e_Y_price(S,K1,K2,T1,T2,r,q,100,sigma))
print("Binomial tree of american compound option:")
print(Y_price(S,K1,K2,T1,T2,r,q,100,sigma))
print('Trinomial tree of american compound option:')
print(C_TT_Am_Y(S, K1, K2, T1, T2, r ,q, 100, sigma))
print('Monte Carlo Simulation of european compound option:')
print(Y_price_simulations(S,r,q,T1,T2,sigma,K1,K2,1000))
print('Least Squares Monte Carlo Simulation of american compound option:')
p =Y_MC_American_call_price(r,q,sigma,S,T1,T2,100,1000,K1,K2)
print(p)
