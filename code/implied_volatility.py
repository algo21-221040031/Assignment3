
from derivative_pricing_models import *
import pandas as pd

'''
关于隐含波动率:
以摩根大通2021.11.24
到期日为2021.12.31,2022.4.14,2023.1.21,2024.1.19的各执行价的期权为例
时间年化为：
12.31 0.0986301369863
4.14  0.3835616438356
2023.1.20  1.1534246575342
2024.1.19  2.1506849315068
'''


file_name_list= ['../data/option_jpm_2021.xlsx','../data/option_jpm_2022.xlsx','../data/option_jpm_2023.xlsx','../data/option_jpm_2024.xlsx']
T_list = [0.0986301369863,0.3835616438356,1.1534246575342,2.1506849315068]
FT_tulpe = list(zip(file_name_list,T_list))

def plot_sigma(file_name,T):
    data = pd.read_excel(file_name)
    df = pd.DataFrame(data)
    K = df["K"]
    C = df["卖价"]
    KC_tulpe = list(zip(K,C))
    sigma_list = [bsm_imp_vol(S,k,T,r,c,q) for (k,c) in KC_tulpe]
    #画出波动率曲线
    plt.figure()
    plt.plot(K, sigma_list, label='sigma')
    
for FT in FT_tulpe:
    file_name = FT[0]
    T = FT[1]
    plot_sigma(file_name,T)
    