import random
from collections import defaultdict
from pathlib import Path
import polyscope as ps
import numpy as np
from community import community_louvain
from math import sqrt
from scipy import fftpack
from sklearn.cluster import SpectralClustering
import scipy
from scipy.sparse import csgraph
from numpy import linalg as LA

import matplotlib.pyplot as plt

import networkx as nx
from pyvis.network import Network


from clustering import read_pointcloud, get_clusters, write_clusters, get_clusters_and_write_to_file, \
    get_clusters_grid_search, ClusterTechnique, EigenVectorClustering, mark_outliers, read_clusters_from_ply, \
    write_clusters_to_ply


def clique_clusters(af):

    nodes = range(0, af.shape[0] - 1)

    #nodes = random.sample(nodes, 1000)

    #net = Network("1024px", "1024px")
    #for n in nodes:
    #    net.add_node(n, label=str(n))
    g = nx.Graph()

    g.add_nodes_from(nodes)


    c = 0;
    for i in nodes:
        for j in nodes:

            if i == j:
                continue

            if af[i, j] == 1:
                g.add_edge(i, j)
                #net.add_edge(i, j)

    print('graph created.')

    #p = community_louvain.best_partition(g)
    #print(p)

    #return p
    c = [] #list(nx.algorithms.community.asyn_lpa_communities(g))

    #net.show("nx.html")



    plt.show()

    return c

def read_sparse_af(file_path, n):
    af = np.zeros((n, n), dtype=int)
    af_marked = np.empty((n, n), dtype=int)
    af_marked.fill(-1)

    f = open(file_path, "r")
    for l in f:
        coords = l.split();
        af[int(coords[0]), int(coords[1])] = int(coords[2])
        af_marked[int(coords[0]), int(coords[1])] = int(coords[2])

    return af, af_marked

if __name__ == '__main__':

    path = Path('C:/') / 'Projekte' / 'csg_playground_build' / 'RelWithDebInfo'
    output_file = 'clusters.dat'

    points, cluster_labels = read_clusters_from_ply('C:/Users/friedrich/Downloads/experiments (1)/experiments/fig4PolyFit/convex_clusters.ply')

    ps.init()

    pc = ps.register_point_cloud("points", points[:, :3], radius=0.005)
    pc.add_scalar_quantity("cluster labels", cluster_labels, enabled=False)

    ps.show()


    input_pc = read_pointcloud(path / 'pc_af.dat', delimiter=' ', hasHeader=True)

    n = input_pc.shape[0]
    print('affinity matrix size: ' + str(n))

    print('load affinity matrix from file')
    afm, afm_marked = read_sparse_af(path / 'af.dat', n)


    #plt.matshow(afm)
    #plt.show()

    cluster_labels, num_clusters, cluster_score = \
        get_clusters_grid_search(afm, ClusterTechnique.RW, EigenVectorClustering.KMEANS, [10])#[2,3,3,4,5,6,7,9,10,11,12,13,14,15])


    write_clusters_to_ply('test.ply', input_pc, cluster_labels)

    #cluster_labels, num_clusters, cluster_score = \
    #    get_clusters_grid_search(afm, ClusterTechnique.SYMMETRIC, EigenVectorClustering.DBSCAN, [3, 4, 5], eps=0.2)

    print('Best cluster score with {} clusters: {}'.format(num_clusters, cluster_score))

    #write_clusters(output_file, input_pc, cluster_labels)


    ps.init()

    pc = ps.register_point_cloud("points", input_pc[:, :3], radius=0.005)

    pc.add_scalar_quantity("cluster labels", cluster_labels, enabled=False)
    pc.add_scalar_quantity("outlier labels", mark_outliers(cluster_labels, input_pc[:, :3]), enabled=False)

    # Clique cluster visualization.
    # pc.add_scalar_quantity("clique labels", com_labels, enabled=False)

    ps.show()
