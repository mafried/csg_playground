import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph
from numpy import linalg as LA
from pathlib import Path
import os.path
import time

def write_clusters(path, pc, cluster_labels):

    num_clusters = len(np.unique(cluster_labels, axis=0).astype(int))

    file = open(path, "w")
    file.write(str(num_clusters) + '\n')

    clusters = {}

    for i, cluster_label in enumerate(cluster_labels):
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(list(pc[i]))

    for cluster in clusters.values():
        np.savetxt(file, np.array(cluster), header=str(len(cluster)) + ' ' + str(6), comments='')

    file.close()

def write_cluster_labels(path, cluster_labels):
    num_clusters = len(np.unique(cluster_labels, axis=0).astype(int))

    file = open(path, "w")
    file.write(str(num_clusters) + '\n')

    for l in cluster_labels:
        file.write(str(l) + '\n')


def get_clusters(affinity_matrix, max_eigval):

	
    topK = 1
    L = csgraph.laplacian(affinity_matrix, normed=True)
    n_components = affinity_matrix.shape[0]
    eigenvalues, eigenvectors = LA.eig(L)

    # for ev in eigenvalues:
    #    print(ev)

    # Sort the eigenvalues (eig doesn't guarantee it)
    eigval = [np.real(ev) for ev in eigenvalues]
    eigval.sort()
    eigval = np.array(eigval)

    # for ev in eigval:
    #    print(ev)
    # np.savetxt('eigval.csv', eigval, delimiter=',')

    # count the number of eigen values less than max_eigval
    count = 0
    for ev in eigval:
        if ev > max_eigval:
            break
        else:
            count = count + 1
	
	
    nb_clusters = count

    # index_largest_gap = np.argsort(np.diff(eigval))[::-1][:topK]
    # nb_clusters = index_largest_gap + 1
    # nb_clusters = nb_clusters[0]

    print('Number clusters: ')
    print(nb_clusters)
    # End of determining number of cluster

    sc = SpectralClustering(n_clusters=nb_clusters, affinity='precomputed')

    sc.fit_predict(affinity_matrix)

    print(sc.labels_.shape)

    return sc.labels_


def read_pointcloud(path, delimiter=' ', hasHeader=True):
    with open(path, 'r') as f:
        if hasHeader:
            # Get rid of the Header
            _ = f.readline()
        # This iterates over all lines, splits them and converts values to floats. Will fail on wrong values.
        pc = [[float(x) for x in line.rstrip().split(delimiter)] for line in f if line != '']

    return np.asarray(pc)[:, :6]


def read_sparse_af(file_path, n):
    af = np.zeros((n, n))

    f = open(file_path, "r")
    for l in f:
        coords = l.split();
        af[int(coords[0]), int(coords[1])] = 1.0

    return af
	
def get_clusters_and_write_to_file(afm_path, n_str, cluster_file, cluster_param_str):
    n = int(n_str)
    cluster_param = float(cluster_param_str)
	
    print('affinity matrix size: ' + str(n))
    print('cluster param: ' + str(cluster_param))

    print('load affinity matrix from file')
    afm = read_sparse_af(afm_path, n)
    print(afm)

    cluster_labels = get_clusters(afm, cluster_param)

    write_cluster_labels(cluster_file, cluster_labels)

    while not os.path.exists(cluster_file):
        print('file not yet there...')
        time.sleep(1)
    print('file available.')
