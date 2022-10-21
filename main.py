from importlib import import_module
import numpy as np
import matplotlib.pyplot as plt

n = 8
r = 10
handles = [2, 3, 6]


def get_ls(xs, ys):
    return np.array([np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1]) ** 2)
                     for i in range(len(xs)) if i != 0])


def get_ns(xs, ys):
    ls = get_ls(xs, ys)
    nxs = np.array([(ys[i] - ys[i-1]) / ls[i-1]
                   for i in range(len(xs)) if i != 0])
    nys = np.array([(-xs[i] + xs[i-1]) / ls[i-1]
                   for i in range(len(xs)) if i != 0])
    return nxs, nys


def get_rad(u, v):
    i = np.inner(u, v)
    n = np.linalg.norm(u) * np.linalg.norm(v)
    c = i / n
    return np.arccos(np.clip(c, -1.0, 1.0))


def opt_v(xs, ys, ls, xs_0, ys_0, ls_0):
    l_avg = sum(ls_0) / len(ls_0)
    nxs, nys = get_ns(xs, ys)

    a_normal = np.array([np.zeros(2 * (n + 1)) for _ in range(2 * n)])
    b_normal = np.zeros(2 * n)
    a_edge = np.array([np.zeros(2 * (n + 1)) for _ in range(2 * n)])
    b_edge = np.zeros(2 * n)

    for i in range(n):
        j = i + 1
        c = max(1, ls[i]/l_avg)

        # set x
        a_normal[i][i] = nxs[i] * c / ls[i]
        a_normal[i][j] = -c * nxs[i] / ls[i]
        a_edge[i][i] = c / ls[i]
        a_edge[i][j] = -c / ls[i]
        b_edge[i] = -c / ls_0[i] * (xs_0[j] - xs_0[i])

        # set y
        a_normal[n + i][n + 1 + i] = nys[i] * c / ls[i]
        a_normal[n + i][n + 1 + j] = -c * nys[i] / ls[i]
        a_edge[n + i][n + 1 + i] = c / ls[i]
        a_edge[n + i][n + 1 + j] = -c / ls[i]
        b_edge[n + i] = -c / ls_0[i] * (ys_0[j] - ys_0[i])

    wc = 100000
    a_handles = [[wc if (i in handles or i-n-1 in handles) and i ==
                  j else 0 for i in range(2 * (n+1))] for j in range(2 * (n+1))]
    px = xs_0
    py = ys_0
    b_handles = np.zeros(2 * (n + 1))
    for i in range(n+1):
        if i in handles:
            b_handles[i] = wc * px[i]
            b_handles[n+1+i] = wc * py[i]

    A_v = np.concatenate([a_normal, a_edge, a_handles])
    b_v = np.concatenate([b_normal, b_edge, b_handles])
    X_v = np.linalg.solve(np.dot(A_v.T, A_v), np.dot(A_v.T, b_v))
    pn = int(len(X_v) / 2)
    xs = X_v[:pn]
    ys = X_v[pn:]
    return xs, ys


def opt_l(xs, ys):
    ls_0 = get_ls(xs, ys)
    l_avg = sum(ls_0) / len(ls_0)
    a_linearized = [
        [1/np.sqrt(l_avg) if i == j else 0 for i in range(n)] for j in range(n)]
    b_linearized = [l / np.sqrt(l_avg) for l in ls_0]
    a_tangent = np.array([np.zeros(n) for _ in range(n)])
    b_tangent = np.zeros(n)

    for i in range(n):
        j = i - 1
        k = (i + 1) % n
        u = np.array([xs[i] - xs[j], ys[i] - ys[j]])
        v = np.array([xs[k] - xs[i], ys[k] - ys[i]])
        rad = get_rad(u, v)
        if rad > np.pi * 95 / 180:
            wt = np.exp(-(rad - np.pi) ** 2 / (2 * (np.pi / 6) ** 2))
        else:
            wt = 0.001
        r = ls_0[i] / ls_0[i-1]
        a = np.sqrt(wt * l_avg / ls_0[i])
        a_tangent[i][i] = a
        a_tangent[i][j] = -a * r

    A_l = np.concatenate([a_linearized, a_tangent])
    b_l = np.concatenate([b_linearized, b_tangent])
    X_l = np.linalg.solve(np.dot(A_l.T, A_l), np.dot(A_l.T, b_l))
    return X_l


xs_0 = np.array([r * np.cos(2 * np.pi * x / n) for x in range(n+1)])
ys_0 = np.array([r * np.sin(2 * np.pi * x / n) for x in range(n+1)])
ls_0 = get_ls(xs_0, ys_0)
xs, ys = xs_0, ys_0
ls = ls_0
for _ in range(10):
    xs, ys = opt_v(xs, ys, ls, xs_0, ys_0, ls_0)
    ls = opt_l(xs, ys)
plt.plot(xs, ys)
plt.axis('square')
plt.show()
