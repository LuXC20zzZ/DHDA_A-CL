from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
import torch
import torch.nn.functional as F
import models
import numpy as np


def RUL_Score(y_pre, y_true):

    y_true = y_true.view(-1).cpu().detach().numpy()
    y_pre = y_pre.view(-1).cpu().detach().numpy()
    d = y_pre - y_true
    mse = np.mean(np.square(d))
    phm_score = np.sum(np.exp(-d[d < 0] / 13) - 1) + np.sum(np.exp(d[d >= 0] / 10) - 1)

    return mse, phm_score




