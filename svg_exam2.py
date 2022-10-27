import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import svgpathtools as svg
import random


def get_ls(xs, ys):
    return np.array([np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1]) ** 2)
                     for i in range(len(xs))])


def get_normals(xs, ys):
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
path_ids = []
xs, ys = [], []
for i, path in enumerate(paths):
    path_xs, path_ys = [], []
    last_x, last_y = None, None
    path_id = 0
    path_n = 0
    if len(path_ids) > 0:
        path_id = path_ids[-1] + 1
    for j, seg in enumerate(path):
        n_sampling = int(ls[i][j] / l_sampling) + 1
        tvals = np.linspace(0, 1, n_sampling)
        seg_ps = seg.poly()(tvals)
        seg_xs = [p.real for p in seg_ps]
        seg_ys = [-p.imag for p in seg_ps]
        if last_x is not None and last_y is not None and abs(last_x - seg_xs[0]) < 1 and abs(last_y - seg_ys[0]) < 1:
            path_xs.pop()
            path_ys.pop()
        elif last_x is not None and last_y is not None:
            path_ids += [path_id for _ in range(len(path_xs))]
            xs += path_xs
            ys += path_ys
            path_id += 1
            path_xs, path_ys = [], []
        path_xs += seg_xs
        path_ys += seg_ys
        last_x = seg_xs[-1]
        last_y = seg_ys[-1]
    xs += path_xs
    ys += path_ys
    path_ids += [path_id for _ in range(len(path_xs))]


fig = plt.figure()
ax1 = fig.add_subplot(111)

# corner detection
corner_ids = []
path_n = path_ids[-1] + 1
for path_id in range(path_n):
    edge_ids = [i for i, id in enumerate(path_ids) if id == path_id]
    for i, current in enumerate(edge_ids):
        j = edge_ids[i - 1]
        k = edge_ids[(i + 1) % len(edge_ids)]
        u = np.array([xs[current] - xs[j], ys[current] - ys[j]])
        v = np.array([xs[k] - xs[current], ys[k] - ys[current]])
        rad = get_rad(u, v)
        if rad > np.pi * 45 / 180:
            corner_ids.append(current)
    plt_xs = [xs[i] for i in edge_ids]
    plt_ys = [ys[i] for i in edge_ids]
    ax1.scatter(plt_xs, plt_ys)


xs_0 = np.array(xs)
ys_0 = np.array(ys)
ls_0 = get_ls(xs_0, ys_0)
handles = corner_ids
plt_xs_h = [x for i, x in enumerate(xs) if i in handles]
plt_ys_h = [y for i, y in enumerate(ys) if i in handles]

# ax1.scatter(xs_0, ys_0, c='k')
ax1.scatter(plt_xs_h, plt_ys_h, c='r')
plt.show()
