from pathlib import Path
import polyscope as ps
import numpy as np
from math import sqrt
from sklearn.cluster import SpectralClustering
import scipy
from scipy.sparse import csgraph
from numpy import linalg as LA

import networkx as nx

from clustering import read_pointcloud, get_clusters, write_clusters, get_clusters_and_write_to_file


def clique_clusters(af):

    g = nx.Graph()

    nodes = range(0, af.shape[0]-1)
    g.add_nodes_from(nodes)

    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if af[i, j] == 1:
                g.add_edge(i, j)

    return list(nx.find_cliques(g))




if __name__ == '__main__':

    path = Path('C:/') / 'Projekte' / 'csg_playground_build' / 'RelWithDebInfo'
    output_file = 'clusters.dat'

    get_clusters_and_write_to_file(path / 'af.dat', '2225')

    '''
    input_pc = read_pointcloud(path / 'pc_af.dat', delimiter=' ', hasHeader=True)

    n = input_pc.shape[0]
    print('affinity matrix size: ' + str(n))

    print('load affinity matrix from file')
    afm = np.reshape(np.fromfile(path / 'af.dat', sep=' '), (n, n))
    print(afm)

    cluster_labels = get_clusters(afm, 0.4)

    # Clique creation.
    # cliques = clique_clusters(afm)
    # print('Cliques: ' + str(len(cliques)))
    # clique_labels = np.zeros(n)
    # for clique_idx, clique in enumerate(cliques):
    #    for point_idx in clique:
    #        clique_labels[point_idx] = clique_idx
   
    write_clusters(output_file, input_pc, cluster_labels)
    
    ps.init()

    pc = ps.register_point_cloud("points", input_pc[:, :3], radius=0.0025)

    pc.add_scalar_quantity("rand vecs", cluster_labels, enabled=False)

    # Clique cluster visualization.
    # pc.add_scalar_quantity("clique labels", clique_labels, enabled=True)

    ps.show()
    '''