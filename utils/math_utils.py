import numpy as np
import torch


def MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mape = (np.abs(v_ - v) / np.abs(v) + 1e-5).astype(np.float64)
    # mape = np.where(mape > 5, 0, mape)
    return np.mean(mape, axis)

def MAPE10(v, v_, axis=None):
    '''
    Mean absolute percentage error for target value no less than 10.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE10 averages on all elements of input.
    '''
    v0 = v.reshape(-1)
    v_0 = v_.reshape(-1)
    index = v0>=10
    v0 = v0[index]
    v_0 = v_0[index]
    mape10 = (np.abs(v_0 - v0) / np.abs(v0 + 1e-5)).astype(np.float64)
    mape10 = np.where(mape10 > 0.6, 0, mape10)
    return np.mean(mape10, axis)


def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)


def evaluate(y, y_hat, by_step=False, by_node=False, mape10=True):
    '''
    :param y: array in shape of [count, time_step, node]. GT
    :param y_hat: in same shape with y. Pred
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :param threshold: bool, if mape_10 is calculated. 
    :return: array of mape, mae and rmse.
    '''
    if mape10:
        if not by_step and not by_node:
            return MAPE10(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
        elif by_step and by_node:
            return MAPE10(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
        elif by_step:
            return MAPE10(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
        elif by_node:
            return MAPE10(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))
    else:
        if not by_step and not by_node:
            return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
        elif by_step and by_node:
            return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
        elif by_step:
            return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
        elif by_node:
            return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))


def creatMask(x):
    res = x
    b, l, c = res.shape
    mask_ratio = torch.nn.Dropout(p=0.2)
    Mask = torch.ones(b, l, c, device=x.device)
    Mask = mask_ratio(Mask)
    Mask = Mask > 0  # torch.Size([8, 1, 48, 48])
    Mask = Mask
    # res.masked_fill_(Mask, 0)
    return Mask
    
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

