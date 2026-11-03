import matplotlib.pyplot as plt
import numpy as np

def fdif(v : np.ndarray, n = 1):
    h = 2 * np.pi / len(v) 
    if n == 1:
        return (np.roll(v, -1) - np.roll(v, 1)) / (2 * h)
    
    return fdif(fdif(v), n - 1)


if __name__ == "__main__":
    # KdV equation: u_t + u*u_x + u_xxx = 0
    # u(x, 0) = 3 * A ** 2 * sech(0.5 * A * (x + 2)) ** 2 + 3 * B ** 2 * sech(0.5 * B * (x + 1)) ** 2, A = 25, B = 16
    N = 2048
    dt = 1e-8
    h = 2 * np.pi / N
    x = h * np.arange(-N//2, N//2)

    # initial conditions
    sech = lambda z: 1 / np.cosh(z)
    A, B = 25, 16
    u = 3 * A ** 2 * sech(0.5 * A * (x + 2)) ** 2 + 3 * B ** 2 * sech(0.5 * B * (x + 1)) ** 2

    # KdV equation in time
    def KdV(v):
        return - (v * fdif(v) + fdif(v, 3))

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
            print('Loading')
            U.append(u.copy())
            t.append(n * dt)

    U = np.array(U)
    t = np.array(t)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    for u_i, t_i in zip(U, t):
        ax.plot(x, np.full_like(x, t_i), u_i, c='black', ls = '-', lw=.8)

    ax.set_xlabel('x'), ax.set_ylabel('t'), ax.set_zlabel('u')
    ax.set_box_aspect((1.4, 1.4, .4))
    plt.show()