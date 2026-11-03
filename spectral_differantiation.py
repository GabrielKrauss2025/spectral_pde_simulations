import numpy as np

def dif(v : np.ndarray, n = 1):
    N = len(v)
    k = np.concatenate([
        np.arange(0, N//2),
        np.array([0]),
        np.arange(-N//2 + 1, 0)
    ])
    v_hat = np.fft.fft(v)
    w_hat = (1j * k) ** n * v_hat
    w = np.real(np.fft.ifft(w_hat))
    return w