from scipy.spatial import Delaunay, delaunay_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import svgpathtools as svg

def svg_bbox(file):
    paths, _ = svg.svg2paths(file)
    for path in paths:
        bbox_xs, bbox_ys = [], []
        ol_xs, ol_ys = [], []
        tvals = np.linspace(0, 1, 10)
        last_x, last_y = None, None
        for j, seg in enumerate(path):
            seg_ps = seg.poly()(tvals)
            seg_xs = [p.real for p in seg_ps]
            seg_ys = [p.imag for p in seg_ps]
            if j > 0 and abs(last_x - seg_xs[0]) > 1 and abs(last_y - seg_ys[0]) > 1:
                bbox_xs.append([min(ol_xs), max(ol_xs), max(ol_xs), min(ol_xs)])
                bbox_ys.append([min(ol_ys), min(ol_ys), max(ol_ys), max(ol_ys)])
                ol_xs, ol_ys = [], []
            ol_xs += seg_xs
            ol_ys += seg_ys
            last_x, last_y = seg_xs[-1], seg_ys[-1]
        bbox_xs.append([min(ol_xs), max(ol_xs), max(ol_xs), min(ol_xs)])
        bbox_ys.append([min(ol_ys), min(ol_ys), max(ol_ys), max(ol_ys)])
    return bbox_xs, bbox_ys

w = h = 360
n = 6
np.random.seed(0)
pts = np.random.randint(0, w, (n, 2))
paths, _ = svg.svg2paths('./はさみのフリーアイコン.svg')
tvals = np.linspace(0, 1, 10)
pts = []
for i, path in enumerate(paths):
    for j, seg in enumerate(path):
        seg_ps = seg.poly()(tvals)
        seg_ps = [[p.real, p.imag] for p in seg.poly()(tvals)]
        pts += seg_ps
pts = np.array(pts)
xs = pts[:, 0]
ys = pts[:, 1]
bbox_xs, bbox_ys = svg_bbox('./はさみのフリーアイコン.svg')
bbox_pts = []
for i in range(len(bbox_xs)):
    for j in range(4):
        bbox_pts.append([bbox_xs[i][j], bbox_ys[i][j]])
bbox_pts = np.array(bbox_pts)
# pts = np.array(pts)
tri = Delaunay(bbox_pts)
fig = delaunay_plot_2d(tri)
plt.scatter(xs, ys)
for i in range(len(bbox_xs)):
    plt.plot(bbox_xs[i] + [bbox_xs[i][0]] , bbox_ys[i] + [bbox_ys[i][0]])
plt.show()
