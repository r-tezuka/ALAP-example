import numpy as np
import svgpathtools as svg
import cProfile
import pstats
import tkinter
from tkinter import ttk
from scipy.spatial import Delaunay, delaunay_plot_2d

def click(event):
    global figure
    global before_x, before_y
    x = event.x
    y = event.y
    figure = canvas.find_closest(x, y)
    before_x = x
    before_y = y

def drag(event):
    global before_x, before_y
    x = event.x
    y = event.y
    canvas.move(
        figure,
        x - before_x, y - before_y
    )
    before_x = x
    before_y = y

def draw(vs, handles):
    r = 3
    _ = [canvas.create_oval(v[0] - r, v[1] - r, v[0] + r, v[1] + r, fill = 'black', outline='') for v in vs]
    r = 6
    for i in handles:
        v = vs[i]
        p = canvas.create_oval(v[0] - r, v[1] - r, v[0] + r, v[1] + r, fill = 'red', outline='', tag='handle')
        canvas.tag_bind(p, "<ButtonPress-1>", click)
        canvas.tag_bind(p, "<Button1-Motion>", drag)

def get_ls(vs, path_starts):
    ls = []
    for i, start in enumerate(path_starts):
        if start == path_starts[-1]:
            end = len(vs)
        else:
            end = path_starts[i+1]
        path_vs = vs[start:end]
        path_ls = [np.linalg.norm(v - path_vs[j-1]) for j, v in enumerate(path_vs)]
        ls += path_ls
    return np.array(ls)


def get_normals(vs, path_starts):
    normals = []
    ls = get_ls(vs, path_starts)
    for i, start in enumerate(path_starts):
        if start == path_starts[-1]:
            end = len(vs)
        else:
            end = path_starts[i+1]
        path_vs = vs[start:end]
        path_ls = ls[start:end]
        path_ns = [[(path_vs[j][1] - path_vs[j-1][1]) / l, (-path_vs[j][0] + path_vs[j-1][0]) / l] for j, l in enumerate(path_ls)]
        normals += path_ns
    return np.array(normals)


def get_rad(u, v):
    i = np.inner(u, v)
    n = np.linalg.norm(u) * np.linalg.norm(v)
    c = i / n
    return np.arccos(np.clip(c, -1.0, 1.0))


def opt_v(ls, vs_0, ls_0, a_handles, b_handles, a_comp, b_comp, path_starts):
    l_avg = sum(ls) / len(ls)
    normals = get_normals(vs_0, path_starts)
    n = len(ls)
    a_normal = np.array([np.zeros(2 * n) for _ in range(n)])
    b_normal = np.zeros(n)
    a_edge = np.array([np.zeros(2 * n) for _ in range(2 * n)])
    b_edge = np.zeros(2 * n)
    for i in range(n):
        if i in path_starts:
            if i == path_starts[-1]:
                j = n - 1
            else:
                j = path_starts[path_starts.index(i)+1] - 1
        else:
            j = i - 1
        c = max([1, ls[i]/l_avg])

        # set x
        a_normal[i][i] = c * normals[i][0] / ls[i]
        a_normal[i][j] = -c * normals[i][0] / ls[i]
        a_edge[i][i] = c / ls[i]
        a_edge[i][j] = -c / ls[i]
        b_edge[i] = c / ls_0[i] * (vs_0[i][0] - vs_0[j][0])

        # set y
        a_normal[i][n + i] = c * normals[i][1] / ls[i]
        a_normal[i][n + j] = -c * normals[i][1] / ls[i]
        a_edge[n + i][n + i] = c / ls[i]
        a_edge[n + i][n + j] = -c / ls[i]
        b_edge[n + i] = c / ls_0[i] * (vs_0[i][1] - vs_0[j][1])

    A_v = np.concatenate([a_normal, a_edge, a_handles, a_comp])
    b_v = np.concatenate([b_normal, b_edge, b_handles, b_comp])
    inv = np.linalg.pinv(np.dot(A_v.T, A_v))
    Atb = np.dot(A_v.T, b_v)
    X_v = np.dot(inv, Atb)
    # X_v = np.linalg.solve(np.dot(A_v.T, A_v), np.dot(A_v.T, b_v))
    pn = int(len(X_v) / 2)
    xs = X_v[:pn]
    ys = X_v[pn:]
    vs = np.array([[xs[i], ys[i]] for i in range(pn)])
    return vs

def opt_l(vs, path_starts):
    ls = get_ls(vs, path_starts)
    l_avg = sum(ls) / len(ls)
    n = len(vs)
    a_linearized = [
        [1/np.sqrt(l_avg) if i == j else 0 for i in range(n)] for j in range(n)]
    b_linearized = [l / np.sqrt(l_avg) for l in ls]
    a_tangent = np.array([np.zeros(n) for _ in range(n)])
    b_tangent = np.zeros(n)

    for m, start in enumerate(path_starts):
        if start == path_starts[-1]:
            end = len(vs)
        else:
            end = path_starts[m+1]
        path_vs = vs[start:end]
        path_ls = ls[start:end]
        n = len(path_vs)
        for i in range(n):
            j = i - 1
            k = (i + 1) % n
            u = path_vs[i] - path_vs[j]
            v = path_vs[k] - path_vs[i]
            rad = get_rad(u, v)
            if rad > np.pi * 95 / 180:
                wt = np.exp(-(rad - np.pi) ** 2 / (2 * (np.pi / 6) ** 2))
            else:
                wt = 0.001
            r = path_ls[i] / path_ls[k]
            a = np.sqrt(wt * l_avg / path_ls[i])
            a_tangent[i + start][i + start] = a
            if j < 0:
                j = n - 1
            a_tangent[i + start][j + start] = -a * r

    A_l = np.concatenate([a_linearized, a_tangent])
    b_l = np.concatenate([b_linearized, b_tangent])
    inv = np.linalg.pinv(np.dot(A_l.T, A_l))
    Atb = np.dot(A_l.T, b_l)
    X_l = np.dot(inv, Atb)

    # X_l = np.linalg.solve(np.dot(A_l.T, A_l), np.dot(A_l.T, b_l))
    return X_l

def main():
    # init
    paths, _ = svg.svg2paths('./はさみのフリーアイコン.svg')
    v_min = [float('inf'), float('inf')]
    v_max = [-float('inf'), -float('inf')]
    segs, ls = [], []
    for i, path in enumerate(paths):
        path_segs, path_ls = [], []
        tvals = np.linspace(0, 1, 10)
        v_last = [None, None]
        for j, seg in enumerate(path):
            seg_vs = np.array([[p.real, p.imag] for p in seg.poly()(tvals)])
            v_min = [min([v_min[0], min(seg_vs[:, 0])]), min([v_min[1], min(seg_vs[:, 1])])]
            v_max = [max([v_max[0], max(seg_vs[:, 0])]), max([v_max[1], max(seg_vs[:, 1])])]
            if j > 0 and abs(v_last[0] - seg_vs[0][0]) > 1 and abs(v_last[1] - seg_vs[0][1]) > 1:
                segs.append(path_segs)
                ls.append(path_ls)
                path_segs, path_ls = [], []
            path_segs.append(seg)
            seg_l = sum([np.linalg.norm(v - seg_vs[i-1]) for i, v in enumerate(seg_vs) if i != 0])
            path_ls.append(seg_l)
            v_last = seg_vs[-1]
        segs.append(path_segs)
        ls.append(path_ls)
    l_sampling = max(np.array(v_max) - np.array(v_min)) * 0.005

    # resampling
    vs = []
    for i, path_segs in enumerate(segs):
        path_vs = []
        for j, seg in enumerate(path_segs):
            n_sampling = int(ls[i][j] / l_sampling) + 1
            tvals = np.linspace(0, 1, n_sampling)
            seg_vs = [[p.real, p.imag] for p in seg.poly()(tvals)][1:]
            path_vs += seg_vs
        vs.append(np.array(path_vs))
    # corner detection
    cids = []
    for path_vs in vs:
        path_cids = []
        for i, v in enumerate(path_vs):
            j = i - 1
            k = (i + 1) % len(path_vs)
            u = v - path_vs[j]
            p = path_vs[k] - v
            rad = get_rad(u, p)
            if rad > np.pi * 45 / 180:
                path_cids.append(i)
        cids.append(path_cids)

    # set initial data for optimization
    vs_0 = []
    path_starts = []
    for i, path_vs in enumerate(vs):
        path_starts.append(len(vs_0))
        vs_0 += list(path_vs)
    path_starts.append(len(vs_0))
    vs_0 = np.array(vs_0)
    bbox_vs = np.array([[min(vs_0[:, 0]), min(vs_0[:, 1])], [max(vs_0[:, 0]),min(vs_0[:, 1])], [max(vs_0[:, 0]), max(vs_0[:, 1])], [min(vs_0[:, 0]), max(vs_0[:, 1])]])
    vs_0 = np.concatenate([vs_0, bbox_vs])
    ls_0 = get_ls(vs_0, path_starts)
    handles = []
    for i, ids in enumerate(cids):
        for id in ids:
            handles.append(id + path_starts[i])

    n = len(vs_0)
    ls = ls_0
    
    # set E comp
    tri = Delaunay(vs_0)
    d_edges = []
    for tri in tri.simplices:
        for i, start in enumerate(tri):
            end = tri[i - 1]
            if [start, end] not in d_edges and [end, start] not in d_edges:
                d_edges.append([start, end])
    # for de in d_edges:
    #     sv = vs_0[de[0]]
    #     ev = vs_0[de[1]]
    #     canvas.create_line(sv[0], sv[1], ev[0], ev[1])
    w_comp = np.sqrt(0.01 / len(d_edges))
    a_comp = np.array([np.zeros(2 * n) for _ in range(2 * len(d_edges))])
    b_comp = np.zeros(2 * len(d_edges))
    for i, de in enumerate(d_edges):
        l = np.linalg.norm(vs_0[de[0]] - vs_0[de[1]])
        a_comp[i][de[0]] = w_comp / l
        a_comp[i][de[0] + n] = w_comp / l
        a_comp[i][de[1]] = -w_comp / l
        a_comp[i][de[1] + n] = -w_comp / l

        b_comp[i] = w_comp * (vs_0[de[0]][0] - vs_0[de[1]][0]) / l
        b_comp[i + n] = w_comp * (vs_0[de[0]][1] - vs_0[de[1]][1]) / l




    
    def update(ls, vs_0, ls_0, handles, a_comp, b_comp, path_starts):
        ids = canvas.find_withtag('handle')
        vs_h = [[(canvas.coords(i)[0] + canvas.coords(i)[2]) / 2, (canvas.coords(i)[1] + canvas.coords(i)[3]) / 2 ] for i in ids]
        wc = np.sqrt(100000)
        a_handles = np.array([[wc if (i in handles or i-n in handles) and i ==
                          j else 0 for i in range(2 * n)] for j in range(2 * n)])
        b_handles = np.zeros(2 * n)
        for i, id in enumerate(handles):
            b_handles[id] = wc * vs_h[i][0]
            b_handles[n+id] = wc * vs_h[i][1]
        for _ in range(5):
            vs = opt_v(ls, vs_0, ls_0, a_handles, b_handles, a_comp, b_comp, path_starts)
            ls = opt_l(vs, path_starts)
        canvas.delete('all')
        draw(vs, handles)


    # init canvas
    global canvas
    root = tkinter.Tk()
    canvas = tkinter.Canvas(
        root,
        width=600, height=600,
        highlightthickness=0,
        bg="white"
    )
    canvas.grid(row=0, column=0)
    draw(vs_0, handles)

    # init button
    button1 = ttk.Button(
        root,
        text='UPDATE',
        command=lambda:update(ls, vs_0, ls_0, handles, a_comp, b_comp, path_starts))
    button1.grid(row=1, column=0)
    root.mainloop()

if __name__ == '__main__':
    cProfile.run('main()', filename='./main.prof')
    sts = pstats.Stats('main.prof')
    sts.sort_stats('cumtime')
    sts.print_stats(20)