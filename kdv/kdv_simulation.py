import matplotlib.pyplot as plt
import numpy as np
import torch

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


if __name__ == "__main__":
    # KdV equation: u_t + u*u_x + u_xxx = 0
    # u(x, 0) = 3 * A ** 2 * sech(0.5 * A * (x + 2)) ** 2 + 3 * B ** 2 * sech(0.5 * B * (x + 1)) ** 2, A = 25, B = 16
    N = 256
    dt = 1e-6
    h = 2 * np.pi / N
    x = h * np.arange(-N//2, N//2)

    # initial conditions
    sech = lambda z: 1 / np.cosh(z)
    A, B = 25, 16
    u = 3 * A ** 2 * sech(0.5 * A * (x + 2)) ** 2 + 3 * B ** 2 * sech(0.5 * B * (x + 1)) ** 2

    # KdV equation in time
    def KdV(v):
        return - (v * dif(v) + dif(v, 3))

    tmax = .006
    n_plt = int(tmax / (50 * dt))
    nmax = round(tmax / dt)
    U, t = [u.copy()], [0.0]

    for n in range(1, nmax + 1):
        a = KdV(u) * dt
        b = KdV(u + .5 * a) * dt
        c = KdV(u + .5 * b) * dt
        d = KdV(u + c) * dt
        u += (a + 2 * b + 2 * c + d) / 6

        if np.mod(n, n_plt) == 0:
            U.append(u.copy())
            t.append(n * dt)

    U = np.array(U)
    t = np.array(t)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    for u_i, t_i in zip(U, t):
        ax.plot(x, np.full_like(x, t_i), u_i, c='black', ls = '-', lw=.8)

    ax.set_xlabel('x'), ax.set_ylabel('t'), ax.set_zlabel('u')
    plt.show()
