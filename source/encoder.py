from numpy import array
from numpy import argmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# define example
data_path = "E:/Masters\Machine Learning Basic Principles/Data Analysis Project/Log-loss/test_data_results_6288.csv";
data = pd.read_csv(data_path, header=None);

log_loss_res = [];
N = len(data);

for i in range(N):
    if data.values[i]==1:
        log_loss_res.append('1,0,0,0,0,0,0,0,0,0');
    elif data.values[i]==2:
        log_loss_res.append('0,1,0,0,0,0,0,0,0,0');
    elif data.values[i]==3:
        log_loss_res.append('0,0,1,0,0,0,0,0,0,0');
    elif data.values[i]==4:
        log_loss_res.append('0,0,0,1,0,0,0,0,0,0');
    elif data.values[i]==5:
        log_loss_res.append('0,0,0,0,1,0,0,0,0,0');
    elif data.values[i]==6:
        log_loss_res.append('0,0,0,0,0,1,0,0,0,0');
    elif data.values[i]==7:
        log_loss_res.append('0,0,0,0,0,0,1,0,0,0');
    elif data.values[i]==8:
        log_loss_res.append('0,0,0,0,0,0,0,1,0,0');
    elif data.values[i]==9:
        log_loss_res.append('0,0,0,0,0,0,0,0,1,0');
    elif data.values[i]==10:
        log_loss_res.append('0,0,0,0,0,0,0,0,0,1');

res = np.asarray(log_loss_res);
res.shape = (N,1);

df = pd.DataFrame(res);
df.to_csv('Submission_LogLoss.csv', index=False, header=False);
