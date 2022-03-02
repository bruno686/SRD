# @File   : load_data.py
# @Author : He Zhuangzhuang
# @Version: 1.0
# @Date   :2022/2/27,下午2:53
import pandas as pd
import numpy as np
from config import get_params
params = vars(get_params())

def load_from_numpy():
    train = np.load('./YelpCHI/Train.npy')
    train_score = np.load('./YelpCHI/Train_Score.npy')
    train_score = train_score.reshape(train_score.shape[0],1)
    train = np.concatenate((train,train_score),axis=1)
    test = np.load('./YelpCHI/Test.npy')
    test_score = np.load('./YelpCHI/Test_Score.npy')
    test_score = test_score.reshape(test_score.shape[0],1)
    test = np.concatenate((test,test_score),axis=1)
    train_pd = pd.DataFrame(train,columns=['userId','itemId','rating'])
    test_pd = pd.DataFrame(test,columns=['userId','itemId','rating'])
    all = pd.concat((train_pd,test_pd),axis=0)
    return train_pd,test_pd,all

def load_data():
    df = pd.read_table(params['inter_path'],sep='\t',names=['userId','itemId','rating','timestamp'])
    return df

if __name__ == "__main__":
    load_from_numpy()

