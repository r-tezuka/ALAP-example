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
segs, ls = [], []
for i, path in enumerate(paths):
    path_segs, path_ls = [], []
    tvals = np.linspace(0, 1, 10)
    last_x, last_y = None, None
    for j, seg in enumerate(path):
        seg_ps = seg.poly()(tvals)
        xs = [p.real for p in seg_ps]
        ys = [-p.imag for p in seg_ps]
        min_x = min([min_x, min(xs)])
        max_x = max([max_x, max(xs)])
        min_y = min([min_y, min(ys)])
        max_y = max([max_y, max(ys)])
        if j > 0 and abs(last_x - xs[0]) > 1 and abs(last_y - ys[0]) > 1:
            segs.append(path_segs)
            ls.append(path_ls)
            path_segs, path_ls = [], []
        path_segs.append(seg)
        seg_l = sum([np.sqrt((xs[i] - xs[i-1]) ** 2 + (ys[i] - ys[i-1]) ** 2)
                     for i in range(len(seg_ps)) if i != 0])
        path_ls.append(seg_l)
        last_x, last_y = xs[-1], ys[-1]
    segs.append(path_segs)
    ls.append(path_ls)
l_sampling = max([max_x - min_x, max_y - min_y]) * 0.005

# resampling
path_ids = []
xs, ys = [], []
for i, path_segs in enumerate(segs):
    path_xs, path_ys = [], []
    for j, seg in enumerate(path_segs):
        n_sampling = int(ls[i][j] / l_sampling) + 1
        tvals = np.linspace(0, 1, n_sampling)
        seg_ps = seg.poly()(tvals)
        seg_xs = [p.real for p in seg_ps]
        seg_ys = [-p.imag for p in seg_ps]
        if j > 0:
            seg_xs = seg_xs[1:]
            seg_ys = seg_ys[1:]
        path_xs += seg_xs
        path_ys += seg_ys
    xs.append(path_xs)
    ys.append(path_ys)

fig = plt.figure()
ax1 = fig.add_subplot(111)

# corner detection
cids = []
for path_id in range(len(xs)):
    path_xs, path_ys = xs[path_id], ys[path_id]
    path_cids = []
    for i in range(len(xs[path_id])):
        j = i - 1
        k = (i + 1) % len(xs[path_id])
        u = np.array([path_xs[i] - path_xs[j], path_ys[i] - path_ys[j]])
        v = np.array([path_xs[k] - path_xs[i], path_ys[k] - path_ys[i]])
        rad = get_rad(u, v)
        if rad > np.pi * 45 / 180:
            path_cids.append(i)
    cids.append(path_cids)
    ax1.scatter(path_xs, path_ys)
    plt_xs_h = [x for i, x in enumerate(path_xs) if i in path_cids]
    plt_ys_h = [y for i, y in enumerate(path_ys) if i in path_cids]
    ax1.scatter(plt_xs_h, plt_ys_h, c='r')
plt.show()
