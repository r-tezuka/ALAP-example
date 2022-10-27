import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import svgpathtools as svg
import random


def get_ls(xs, ys):
    return np.array([np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1]) ** 2)
                     for i in range(len(xs))])


def get_ns(xs, ys):
    ls = get_ls(xs, ys)
    nxs = np.array([(ys[i] - ys[i-1]) / ls[i]
                   for i in range(len(xs))])
    nys = np.array([(-xs[i] + xs[i-1]) / ls[i]
                   for i in range(len(xs))])
    return nxs, nys


def get_rad(u, v):
    i = np.inner(u, v)
    n = np.linalg.norm(u) * np.linalg.norm(v)
    c = i / n
    return np.arccos(np.clip(c, -1.0, 1.0))


# init
paths, attributes = svg.svg2paths('./icon.svg')
min_x, max_x = float('inf'), -float('inf')
min_y, max_y = float('inf'), -float('inf')
ls = [[] for _ in range(len(paths))]
for i, path in enumerate(paths):
    tvals = np.linspace(0, 1, 10)
    path_ps = [seg.poly()(tvals) for seg in path]
    for seg_ps in path_ps:
        xs = [p.real for p in seg_ps]
        ys = [-p.imag for p in seg_ps]
        l = sum([np.sqrt((xs[i] - xs[i-1]) ** 2 + (ys[i] - ys[i-1]) ** 2)
                for i in range(len(seg_ps)) if i != 0])
        ls[i].append(l)
        min_x = min([min_x, min(xs)])
        max_x = max([max_x, max(xs)])
        min_y = min([min_y, min(ys)])
        max_y = max([max_y, max(ys)])
l_sampling = max([max_x - min_x, max_y - min_y]) * 0.005
ns = [[int(l / l_sampling) + 1 for l in path_ls] for path_ls in ls]

# resampling
path_ns = []
xs, ys = [], []
corner_ids = []
for i, path in enumerate(paths):
    path_xs, path_ys = [], []
    edge_ids = []
    for j, seg in enumerate(path):
        n_sampling = int(ls[i][j] / l_sampling) + 1
        tvals = np.linspace(0, 1, n_sampling)
        seg_ps = seg.poly()(tvals)
        seg_xs = [p.real for p in seg_ps]
        seg_ys = [-p.imag for p in seg_ps]
        edge_ids.append(len(path_xs))
        last = len(seg_xs) - 1
        path_xs += seg_xs[:last]
        path_ys += seg_ys[:last]

    # corner detection
    n = len(path_xs)
    for current in edge_ids:
        j = current - 1
        k = (current + 1) % n
        u = np.array([path_xs[current] - path_xs[j],
                     path_ys[current] - path_ys[j]])
        v = np.array([path_xs[k] - path_xs[current],
                     path_ys[k] - path_ys[current]])
        rad = get_rad(u, v)
        if rad > np.pi * 45 / 180:
            corner_ids.append(current + len(xs))
    path_ns.append(len(path_xs))
    xs += path_xs
    ys += path_ys

xs_0 = np.array(xs)
ys_0 = np.array(ys)
ls_0 = get_ls(xs_0, ys_0)
handles = corner_ids


def opt_v(xs, ys, ls, xs_0, ys_0, ls_0, xs_h, ys_h):
    l_avg = sum(ls) / len(ls)
    nxs, nys = get_ns(xs_0, ys_0)

    a_normal = np.array([np.zeros(2 * n) for _ in range(n)])
    b_normal = np.zeros(n)
    a_edge = np.array([np.zeros(2 * n) for _ in range(2 * n)])
    b_edge = np.zeros(2 * n)

    for i in range(n):
        j = i - 1
        if j < 0:
            j = n - 1
        c = max([1, ls[i]/l_avg])

        # set x
        a_normal[i][i] = c * nxs[i] / ls[i]
        a_normal[i][j] = -c * nxs[i] / ls[i]
        a_edge[i][i] = c / ls[i]
        a_edge[i][j] = -c / ls[i]
        b_edge[i] = c / ls_0[i] * (xs_0[i] - xs_0[j])

        # set y
        a_normal[i][n + i] = c * nys[i] / ls[i]
        a_normal[i][n + j] = -c * nys[i] / ls[i]
        a_edge[n + i][n + i] = c / ls[i]
        a_edge[n + i][n + j] = -c / ls[i]
        b_edge[n + i] = c / ls_0[i] * (ys_0[i] - ys_0[j])

    wc = 100000
    a_handles = [[wc if (i in handles or i-n in handles) and i ==
                  j else 0 for i in range(2 * n)] for j in range(2 * n)]

    b_handles = np.zeros(2 * n)
    for i in range(n):
        if i in handles:
            b_handles[i] = wc * xs_h[i]
            b_handles[n+i] = wc * ys_h[i]

    A_v = np.concatenate([a_normal, a_edge, a_handles])
    b_v = np.concatenate([b_normal, b_edge, b_handles])
    A_v = np.concatenate([a_normal, a_edge, a_handles])
    b_v = np.concatenate([b_normal, b_edge, b_handles])

    X_v = np.linalg.solve(np.dot(A_v.T, A_v), np.dot(A_v.T, b_v))
    pn = int(len(X_v) / 2)
    xs = X_v[:pn]
    ys = X_v[pn:]
    return xs, ys


def opt_l(xs, ys):
    ls = get_ls(xs, ys)
    l_avg = sum(ls) / len(ls)
    a_linearized = [
        [1/np.sqrt(l_avg) if i == j else 0 for i in range(n)] for j in range(n)]
    b_linearized = [l / np.sqrt(l_avg) for l in ls]
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
        r = ls[i] / ls[k]
        a = np.sqrt(wt * l_avg / ls[i])
        a_tangent[i][i] = a
        a_tangent[i][j] = -a * r

    A_l = np.concatenate([a_linearized, a_tangent])
    b_l = np.concatenate([b_linearized, b_tangent])
    X_l = np.linalg.solve(np.dot(A_l.T, A_l), np.dot(A_l.T, b_l))
    return X_l


ims = []
fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
ax1.set_title('initialize')
ax2.set_title('set handle position')
ax3.set_title('optimize')
im1 = ax1.scatter(xs_0, ys_0, c='k')
plt_xs_h = [x for i, x in enumerate(xs_0) if i in handles]
plt_ys_h = [y for i, y in enumerate(ys_0) if i in handles]
ax1.scatter(plt_xs_h, plt_ys_h, c='r')
ax1.axis('square')

n = len(xs)
xs_h = [50 + xs_0[i]
        if i in [handles[4], handles[5]] else xs_0[i] for i in range(n)]
ys_h = [50 + ys_0[i]
        if i in [handles[4], handles[5]] else ys_0[i] for i in range(n)]

im2 = ax2.scatter(xs_h, ys_h, c='k')
plt_xs_h = [x for i, x in enumerate(xs_h) if i in handles]
plt_ys_h = [y for i, y in enumerate(ys_h) if i in handles]
ax2.scatter(plt_xs_h, plt_ys_h, c='r')
ax2.axis('square')

xs, ys = xs_0, ys_0
ls = ls_0


for _ in range(5):
    xs, ys = opt_v(xs, ys, ls, xs_0, ys_0, ls_0, xs_h, ys_h)
    ls = opt_l(xs, ys)
    im3 = ax3.scatter(xs, ys, c='k')
    plt_xs_h = [x for i, x in enumerate(xs) if i in handles]
    plt_ys_h = [y for i, y in enumerate(ys) if i in handles]
    ax3.scatter(plt_xs_h, plt_ys_h, c='r')
    ax3.axis('square')
    ims.append([im1]+[im2]+[im3])
    # ims.append(im1+im2+im3)
ani = animation.ArtistAnimation(fig, ims, interval=100)
plt.show()
