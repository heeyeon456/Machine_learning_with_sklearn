import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from math import sqrt
from sklearn import metrics


def rmse_cal(target,prediction):
    return float(sqrt(metrics.mean_squared_error(target,prediction)))

def mae_cal(target,prediction):
    return float(metrics.mean_absolute_error(target,prediction))

def cor_cal(target,prediction):
    return pearsonr(target,prediction)

def mean_cal(list,n):
    return list.sum()/n

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump
def median_error(target,prediction):
    return np.median(abs(np.subtract(target,prediction)))


def df_select(input_df,num):
    """random sampling with replacement"""
    df=pd.DataFrame(columns=input_df.columns)
    #df.loc[0]=input_df.iloc[0]
    for i in range(num):
        df = pd.concat([df,input_df.sample(n=1)])
    return df

def accuracy(real,pred):
    return metrics.accuracy_score(real,pred)

def aupr(real,yscore):
    """average precision (AP) from prediction scores"""
    return metrics.average_precision_score(real,yscore)

def f1_score(real,pred):
    return metrics.f1_score(real,pred)

def precision(real,pred):
    return metrics.precision_score(real,pred)

def recall(real,pred):
    return metrics.recall_score(real,pred)

def make_binary(true_label,false_label,y_data):
    """make binary data"""
    y_data.replace([true_label,false_label],[1,0],inplace=True)
    return y_data
