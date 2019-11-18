import numpy as np
from scipy.signal import filtfilt
from scipy.signal import fftconvolve

def SNR_to_var(snr):
    return 1 / 10 ** (snr / 20)

def reverberate_tensor(tensor, rir_tensor):
    res = []
    for i in range(rir_tensor.shape[1]):

        res.append(fftconvolve(tensor, rir_tensor[:, i]))

    return np.vstack(res).T

def WGN(shape, SNR):
    var = SNR_to_var(SNR)

    return np.random.normal(0, var, shape)

def peakNorm(arr, DBFS=0):
    gain = 10 ** (DBFS / 20)

    return arr * (gain * 1 / (np.max(np.abs(arr))+ + np.finfo(float).eps))

def pad_tensor(x, target_len, offset=0):

    f = lambda x: pad(x, target_len, offset)

    return np.apply_along_axis(f, 0, x)


def pad(x, target_len, offset=0):
    if len(x) == target_len:
        return x
    elif len(x) > target_len:
        raise ValueError

    if offset > 0:

        out = np.concatenate([WGN(offset, SNR=90), x])
        if len(out) < target_len:
            out = np.concatenate([out, WGN(target_len - len(out), SNR=90)])
        elif len(out) > target_len:
            out = out[:target_len]

    else:
        out = np.concatenate([x, WGN(target_len - len(x), SNR=90)])
    return out


def filtfilter(data, fir, padding=False):
    orig_len = len(data)
    data = filtfilt(fir, 1, data)

    if padding == True:
        data = pad(data, orig_len)

    return data