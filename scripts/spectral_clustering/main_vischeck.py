from pathlib import Path
import polyscope as ps
import numpy as np
from clustering import read_pointcloud


def read_sparse_af(file_path, n):
    af = np.zeros((n, n))
    af_marked = np.empty((n, n))
    af_marked.fill(-1)

    f = open(file_path, "r")
    for l in f:
        coords = l.split();
        af[int(coords[0]), int(coords[1])] = float(coords[2])
        af_marked[int(coords[0]), int(coords[1])] = float(coords[2])

    return af, af_marked

if __name__ == '__main__':
    # Change this
    path = Path('C:/') / 'Projekte' / 'csg_playground_build' / 'RelWithDebInfo'

    input_pc = read_pointcloud(path / 'pc_af.dat', delimiter=' ', hasHeader=True)

    n = input_pc.shape[0]
    print('affinity matrix size: ' + str(n))

    print('load affinity matrix from file')
    # afm = np.reshape(np.fromfile(path / 'af.dat', sep=' '), (n, n))
    afm, afm_marked = read_sparse_af(path / 'af.dat', n)

    ps.init()

    # source point for the visibility visualization
    src = 1000  # change this to the index of a point in the cloud
    visible_idx = []
    not_visible_idx = []
    ni = afm.shape[0]
    nj = afm.shape[1]
    for j in range(nj):
        if j == src:
            continue
        if afm[src, j] == 1:
            visible_idx.append(j)
        else:
            not_visible_idx.append(j)

    ps.register_point_cloud("visible", input_pc[visible_idx, :3], radius=0.005)
    ps.register_point_cloud("not-visible", input_pc[not_visible_idx, :3], radius=0.005)
    ps.register_point_cloud("source", input_pc[src:(src + 1), :3], radius=0.005)

    ps.show()