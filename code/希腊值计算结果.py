
T1 = 0.1
T2 = 0.3
K1 = 10
K2 = 100
from DerivativesProject import *
'''关于Y的希腊值'''

S_price = range(80,151,5)

# The relationship between delta and stock price
plt.figure()
#ed = [delta(s,K1,K2,T1,T2,r,q,100,sigma)[0] for s in S_price]
ad_01 = [delta(s,K1,K2,0.1,T2,r,q,100,sigma)[1] for s in S_price]
ad_005 = [delta(s,K1,K2,0.05,T2,r,q,100,sigma)[1] for s in S_price]
ad_02 = [delta(s,K1,K2,0.2,T2,r,q,100,sigma)[1] for s in S_price]
ad_03 = [delta(s,K1,K2,0.3,T2,r,q,100,sigma)[1] for s in S_price]
#plt.plot(S_price, ed, label='eur_delta')
plt.plot(S_price, ad_01, label='T1 = 0.1')
plt.plot(S_price, ad_005, label='T1 = 0.05')
plt.plot(S_price, ad_02, label='T1 = 0.2')
plt.plot(S_price, ad_03, label='T1 = 0.3')
#plt.plot(K2, c, label='c')
plt.xlabel('S')
plt.ylabel('delta')
plt.legend(loc=2)

# The relationship between gamma and stock price.
# plt.figure()
# ag_01 = [gamma(s,K1,K2,0.1,T2,r,q,100,sigma)[1] for s in S_price]
# ag_005 = [gamma(s,K1,K2,0.05,T2,r,q,100,sigma)[1] for s in S_price]
# ag_02 = [gamma(s,K1,K2,0.2,T2,r,q,100,sigma)[1] for s in S_price]
# ag_03 = [gamma(s,K1,K2,0.3,T2,r,q,100,sigma)[1] for s in S_price]
# plt.plot(S_price, ag_01, label='T1 = 0.1')
# plt.plot(S_price, ag_005, label='T1 = 0.05')
# plt.plot(S_price, ag_02, label='T1 = 0.2')
# plt.plot(S_price, ag_03, label='T1 = 0.3')
# plt.xlabel('S')
# plt.ylabel('gamma')
# plt.legend(loc=2)

#The relationship between theta and stock price
plt.figure()
at_01 = [theta(s,K1,K2,0.1,T2,r,q,100,sigma)[1] for s in S_price]
at_005 = [theta(s,K1,K2,0.05,T2,r,q,100,sigma)[1] for s in S_price]
at_02 = [theta(s,K1,K2,0.2,T2,r,q,100,sigma)[1] for s in S_price]
at_03 = [theta(s,K1,K2,0.3,T2,r,q,100,sigma)[1] for s in S_price]
plt.plot(S_price, at_01, label='T1 = 0.1')
plt.plot(S_price, at_005, label='T1 = 0.05')
plt.plot(S_price, at_02, label='T1 = 0.2')
plt.plot(S_price, at_03, label='T1 = 0.3')
plt.xlabel('S')
plt.ylabel('theta')
plt.legend(loc=4)

#改进，在二叉树里计算delta：
def n_delta(S,K1,K2,T1,T2,r,q,N,sigma):
    dt = T1/N
    u = math.exp((r-q)*dt + sigma*(dt**0.5))
    d = math.exp((r-q)*dt - sigma*(dt**0.5))
    p = (math.exp((r-q)*dt)-d)/(u-d)
    lat = np.zeros((N+1,N+1))
    #n_lat = np.zeros((N+1,N+1))
    for j in range(0,N+1):
        lat[N,j] = max(0,c_BS(S*(u**j)*(d**(N-j)),K2,T2,r,q,sigma)-K1)
    for i in range(N-1,-1,-1):
        for j in range(0,i+1):
            
            lat[i,j] = max(c_BS(S*(u**j)*(d**(i-j)),K2,T2,r,q,sigma)-K1,\
                           math.exp(-r*dt)*(p* lat[i+1,j+1] + (1-p) * lat[i+1,j]))
            '''
            lat[i,j] = max(e_bina_call(S*(u**j)*(d**(i-j)),K2,T2,r,q,100,sigma)-K1,\
                           math.exp(-r*dt)*(p* lat[i+1,j+1] + (1-p) * lat[i+1,j]))
            '''
    delta = math.exp(-r*dt)*(lat[1,1] -lat[1,0])/(S*u-S*d)
    return delta

def n_gamma(S,K1,K2,T1,T2,r,q,N,sigma):
    d2 = n_delta(S+0.01,K1,K2,T1,T2,r,q,N,sigma)
    d1 = n_delta(S,K1,K2,T1,T2,r,q,N,sigma)
    return (d2-d1)/0.01

#画出更精确的gamma
plt.figure()
ag_01 = [n_gamma(s,K1,K2,0.1,T2,r,q,100,sigma) for s in S_price]
ag_005 = [n_gamma(s,K1,K2,0.05,T2,r,q,100,sigma) for s in S_price]
ag_02 = [n_gamma(s,K1,K2,0.2,T2,r,q,100,sigma) for s in S_price]
ag_03 = [n_gamma(s,K1,K2,0.3,T2,r,q,100,sigma) for s in S_price]
plt.plot(S_price, ag_01, label='T1 = 0.1')
plt.plot(S_price, ag_005, label='T1 = 0.05')
plt.plot(S_price, ag_02, label='T1 = 0.2')
plt.plot(S_price, ag_03, label='T1 = 0.3')
plt.xlabel('S')
plt.ylabel('gamma')
plt.legend(loc=2)   