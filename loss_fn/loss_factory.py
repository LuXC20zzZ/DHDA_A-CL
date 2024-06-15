import torch


def RMSE(outputs, labels):
    loss = torch.sqrt(torch.mean(torch.square(outputs - labels)))
    return loss.mean()


def MLEGLoss(outputs_mu, outputs_std, labels, lam=1e-5):
    loss = torch.log(outputs_std**2+1e-6) / 2 + (labels - outputs_mu) ** 2 / (2 * outputs_std ** 2 + 1e-6) + lam * outputs_std ** 4
    return loss.mean()