from collections import defaultdict
from enum import Enum

import numpy as np
from pyod.models.knn import KNN
from sklearn.cluster import SpectralClustering, DBSCAN
from scipy.sparse import csgraph
from numpy import linalg as LA
from pathlib import Path
import os.path
import time

import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import spdiags

from sklearn.cluster import KMeans


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


class ClusterTechnique(Enum):
    SKLEARN = 0
    RW = 1
    SYMMETRIC = 2

class EigenVectorClustering(Enum):
    KMEANS = 0
    DBSCAN = 1

def get_clusters_rw(affinity_matrix, ev_clustering, **kwargs):
    epsilon = 1e-10

    n = affinity_matrix.shape[0]

    # degree matrix
    S = np.sum(affinity_matrix, axis=1)
    Sdiag = [s for s in S]
    D = spdiags(Sdiag, 0, n, n)

    # inv of degree matrix
    IS = [1 / (s + epsilon) for s in S]
    ID = spdiags(IS, 0, n, n)

    # graph Laplacian
    L = D - affinity_matrix

    # find the eigenvectors of the graph Laplacian
    k = 200  # eigenvalutes to compute
    DiL = ID * L
    EVL, EV = scipy.sparse.linalg.eigsh(DiL, k=k, which='SM')

    # sort the eigenvalues (eigsh doesn't guarantee it)
    eigval = [np.real(ev) for ev in EVL]
    sort_idx = np.argsort(eigval)

    eigval.sort()
    eigval = np.array(eigval)

    nb_clusters = 0
    if 'max_eigval' in kwargs:
        # count the number of eigen values less than max_eigval
        count = 0
        for ev in eigval:
            if ev > kwargs['max_eigval']:
                break
            else:
                count = count + 1
        nb_clusters = count
    else:
        nb_clusters = kwargs['num_clusters']

    EVk = EV[:, sort_idx[1:nb_clusters]]
    # normalize rows of EVk
    nrows = EVk.shape[0]
    for i in range(nrows):
        normEVKi = np.linalg.norm(EVk[i, :])
        EVk[i, :] = EVk[i, :] / (normEVKi + epsilon)

    if ev_clustering == EigenVectorClustering.KMEANS:
        return KMeans(n_clusters=nb_clusters, random_state=0).fit(EVk).labels_
    else:
        return DBSCAN(eps=kwargs['eps']).fit(EVk).labels_


def get_clusters_symmetric(affinity_matrix, ev_clustering, **kwargs):
    epsilon = 1e-10

    n = affinity_matrix.shape[0]

    # degree matrix
    S = np.sum(affinity_matrix, axis=1)
    Sdiag = [s for s in S]
    D = spdiags(Sdiag, 0, n, n)

    # inv of degree matrix
    IS = [1 / (np.sqrt(s) + epsilon) for s in S]
    ID = spdiags(IS, 0, n, n)

    # graph Laplacian
    L = D - affinity_matrix

    # find the eigenvectors of the graph Laplacian
    k = 200  # eigenvalutes to compute
    DiL = ID * L * ID
    EVL, EV = scipy.sparse.linalg.eigsh(DiL, k=k, which='SM')

    # sort the eigenvalues (eigsh doesn't guarantee it)
    eigval = [np.real(ev) for ev in EVL]
    sort_idx = np.argsort(eigval)

    eigval.sort()
    eigval = np.array(eigval)

    nb_clusters = 0
    #if ev_clustering == EigenVectorClustering.KMEANS:
    if 'max_eigval' in kwargs:
        # count the number of eigen values less than max_eigval
        count = 0
        for ev in eigval:
            if ev > kwargs['max_eigval']:
                break
            else:
                count = count + 1
        nb_clusters = count
    else:
        nb_clusters = kwargs['num_clusters']


    EVk = EV[:, sort_idx[1:nb_clusters]]
    # normalize rows of EVk
    nrows = EVk.shape[0]
    for i in range(nrows):
        normEVKi = np.linalg.norm(EVk[i, :])
        EVk[i, :] = EVk[i, :] / (normEVKi + epsilon)

    if ev_clustering == EigenVectorClustering.KMEANS:
        print('Clusters: ' + str(nb_clusters))
        return KMeans(n_clusters=nb_clusters, random_state=0).fit(EVk).labels_
    else:
        db = DBSCAN(eps=kwargs['eps']).fit(EVk)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        return labels
        #return DBSCAN(eps=kwargs['eps']).fit(EVk).labels_


def get_clusters_sklearn(affinity_matrix, **kwargs):
    L = csgraph.laplacian(affinity_matrix, normed=True)
    n_components = affinity_matrix.shape[0]

    # number of eigenvalues to compute
    k = 200
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    # sort the eigenvalues (eig doesn't guarantee it)
    eigval = [np.real(ev) for ev in eigenvalues]
    eigval.sort()
    eigval = np.array(eigval)

    nb_clusters = 0
    if 'max_eigval' in kwargs:
        # count the number of eigen values less than max_eigval
        count = 0
        for ev in eigval:
            if ev > kwargs['max_eigval']:
                break
            else:
                count = count + 1
        nb_clusters = count
    else:
        nb_clusters = kwargs['num_clusters']

    sc = SpectralClustering(n_clusters=nb_clusters, affinity='precomputed')

    sc.fit_predict(affinity_matrix)

    return sc.labels_


def get_clusters(affinity_matrix, cluster_technique, ev_clustering, **kwargs):
   if cluster_technique == ClusterTechnique.RW:
       return get_clusters_rw(affinity_matrix, ev_clustering, **kwargs)
   elif cluster_technique == ClusterTechnique.SYMMETRIC:
       return get_clusters_symmetric(affinity_matrix, ev_clustering, **kwargs)
   elif cluster_technique == ClusterTechnique.SKLEARN:
       return get_clusters_sklearn(affinity_matrix, **kwargs)


def get_cluster_score(cluster_labels, aff_mat, alpha=1.0):

    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)

    score = 0.0

    for cluster in clusters.values():

        intra_visible_pairs = 0.0

        for i in cluster:
            for j in cluster:
                if aff_mat[i, j] == 1:
                    intra_visible_pairs += 1.0

        point_lookup = set(cluster)

        inter_occluded_pairs = 0.0

        n = aff_mat.shape[0]

        for i in cluster:
            for j in range(0, n):
                if j not in point_lookup and aff_mat[i,j] == 0:
                    inter_occluded_pairs += 1.0

        score += intra_visible_pairs + alpha * inter_occluded_pairs

    print("clusters: {} score: {} {} ".format(len(clusters), intra_visible_pairs, inter_occluded_pairs))

    return score #/ n**2 #/ float(len(clusters))**2


def get_clusters_grid_search(afm, cluster_technique, ev_clustering, num_clusters_range, **kwargs):
    best_cluster_labels = None
    best_clustering_score = 0.0
    best_num_clusters = 0

    for num_clusters in num_clusters_range:

        cluster_labels = get_clusters(afm, cluster_technique, ev_clustering, num_clusters=num_clusters, **kwargs) if num_clusters > 1 else np.zeros(afm.shape[0])

        score = get_cluster_score(cluster_labels, afm)

        if score > best_clustering_score:
            best_cluster_labels = cluster_labels
            best_clustering_score = score
            best_num_clusters = num_clusters
            print("Current best: {} with {} clusters.".format(best_clustering_score, best_num_clusters))

    return best_cluster_labels, best_num_clusters, best_clustering_score


def read_pointcloud(path, delimiter=' ', hasHeader=True):
    with open(path, 'r') as f:
        if hasHeader:
            # Get rid of the Header
            _ = f.readline()
        # This iterates over all lines, splits them and converts values to floats. Will fail on wrong values.
        pc = [[float(x) for x in line.rstrip().split(delimiter)] for line in f if line != '']

    return np.asarray(pc)[:, :6]


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

def mark_outliers(cluster_labels, pc):
    clusters = defaultdict(list)
    clusters_to_global_point_idx = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append(pc[i,:3])
        clusters_to_global_point_idx[label].append(i)

    global_outlier_labels = np.zeros((cluster_labels.shape[0]))
    for label in clusters.keys():
        clf = KNN()
        clf.fit(clusters[label])
        outlier_labels = clf.labels_
        for local_label_idx, global_point_idx in enumerate(clusters_to_global_point_idx[label]):
            global_outlier_labels[global_point_idx] = -1 if outlier_labels[local_label_idx] else cluster_labels[global_point_idx]

    return global_outlier_labels

color_table = [
    [0, 46, 255, 255],
    [0, 185, 255, 255],
    [0, 255, 46, 255],
    [0, 255, 185, 255],
    [92, 0, 255, 255],
    [92, 255, 0, 255],
    [231, 0, 255, 255],
    [231, 255, 0, 255],
    [255, 0, 0, 255],
    [255, 0, 139, 255],
    [255, 139, 0, 255]
]

def write_clusters_to_ply(path, pc, cluster_labels):

    file = open(path, "w")
    file.write('ply\n')
    file.write('format ascii 1.0\n')
    file.write('element vertex '+ str(pc.shape[0]) +'\n')
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property float nx\n')
    file.write('property float ny\n')
    file.write('property float nz\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('property uchar alpha\n')
    file.write('end_header\n')

    clusters = {}

    for i, cluster_label in enumerate(cluster_labels):
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(list(pc[i]))

    for i, cluster in enumerate(clusters.values()):
        color = color_table[i % len(color_table)]
        for point in cluster:
            point.extend(color)
            for coord in point:
                file.write('{} '.format(coord))
            file.write('\n')
    file.close()

def read_clusters_from_ply(path):
    with open(path, 'r') as f:

        cluster_ids = {}

        while f.readline() != 'end_header\n':
            continue

        cluster_labels = []
        points = []
        cluster_idx = 0;
        for i, line in enumerate(f):
            values = line.split()

            cluster_id = (float(values[6]), float(values[7]), float(values[8]))
            if cluster_id in cluster_ids:
                mapped_cluster_id = cluster_ids[cluster_id]
            else:
                cluster_idx += 1
                mapped_cluster_id = cluster_idx
                cluster_ids[cluster_id] = cluster_idx

            coords = [float(x) for x in values[:6]]

            points.append(coords)
            cluster_labels.append(mapped_cluster_id)


    return np.asarray(points), np.asarray(cluster_labels)



def get_clusters_and_write_to_file(afm_path, n_str, cluster_file, min_clusters_str, max_clusters_str):
    n = int(n_str)
    min_clusters = int(min_clusters_str)
    max_clusters = int(max_clusters_str)

    print('affinity matrix size: ' + str(n))
    print('min clusters: ' + str(min_clusters))
    print('max clusters: ' + str(max_clusters))

    print('load affinity matrix from file')
    afm, afm_marked = read_sparse_af(afm_path, n)
    print(afm)

    cluster_labels, num_clusters, cluster_score = \
        get_clusters_grid_search(afm, ClusterTechnique.RW, EigenVectorClustering.KMEANS,
                                 range(min_clusters, max_clusters + 1))  # [2,3,3,4,5,6,7,9,10,11,12,13,14,15])

    write_cluster_labels(cluster_file, cluster_labels)

    while not os.path.exists(cluster_file):
        print('file not yet there...')
        time.sleep(1)
    print('file available.')
