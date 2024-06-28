import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.signal as signal
import math
from sklearn.metrics import r2_score


# Acc
def accuracy(y_true, y_pred):
    h1 = -abs(y_pred - y_true)
    h2 = abs(y_true)
    accuracy = np.mean(np.exp(h1 / h2))
    return accuracy


# RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


# MAE
def mae(y_true, y_pred):
    mae = np.mean(abs(y_pred - y_true))
    return mae


# MAPE
def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n
    return mape


# Score
def score_fnc(y_true, y_pred):
    Er = ((y_true - y_pred) / y_true) * 100
    score_sum = 0
    for i in range(0, len(y_true)):
        Er_i = Er.values[i]
        if Er_i <= 0:
            score = math.exp(-math.log(0.5)*(Er_i/5))
        else:
            score = math.exp(math.log(0.5)*(Er_i/20))
        score_sum += score
    score_avg = score_sum / len(y_true)
    return score_avg


def DrawEngine(df, file_name):
    df = df.sort_values(by='label', ascending=False, axis=0)
    true = df['label']
    pred = df['pred']

    NN = len(pred) * 160
    size = math.floor(NN/10000)

    if size % 2 == 0:
        size = size+1
    pred = signal.medfilt(pred, size)

    Lnumber = len(true)
    L = []
    m = 1
    while m <= Lnumber:
        L.append(m/Lnumber)
        m = m + 1
    L = pd.Series(L)

    plt.figure(figsize=(8, 6))

    line1, = plt.plot(L, true, color='r', lw=2)
    line2, = plt.plot(L, pred, color='b', lw=1.5)

    Fontsize = 20
    Family = 'Times New Roman'

    plt.xticks(rotation=20, fontsize=Fontsize,
               # fontproperties=Family
               )
    plt.yticks(rotation=20, fontsize=Fontsize,
               # fontproperties=Family
               )
    plt.xlabel('Normalized Time', fontsize=Fontsize,
               fontproperties=Family
               )
    plt.ylabel('Normalized RUL', fontsize=Fontsize,
               fontproperties=Family
               )

    fontdict = {'family': '',
                'weight': 'normal',
                'size': '18'}

    plt.legend([line1, line2], ["Actual RUL", "Predicted RUL"], loc='lower left', ncol=1, framealpha=0.5, prop=fontdict)

    plt.tight_layout()
    plt.savefig('./fig_Rul_cycle/' + file_name, dpi=600, bbox_inches='tight')
    plt.show()

    Aaccuracy = accuracy(true, pred)
    Rmse = rmse(true, pred)
    Mae = mae(true, pred)
    Mape = mape(true, pred)
    Score = score_fnc(true, pred)
    R2 = r2_score(true, pred)

    print('Acc: {:.4f}'.format(Aaccuracy))
    print('RMSE: {:.4f}'.format(Rmse))
    print('MAE: {:.4f}'.format(Mae))
    print('MAPE: {:.4f}'.format(Mape))
    print('Score: {:.4f}'.format(Score))
    print('R2: {:.4f}'.format(R2))


file_name = 'PHM1-1&1-2toXJTU1-1DHDA_ACL'
print(file_name)
result_dir = './results/' + file_name + '.pkl'
test_pd = pd.read_pickle(result_dir)
DrawEngine(test_pd, file_name)
