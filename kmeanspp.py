"""
Authors: Jun Wang
         Farhad Mohsin
         Shaunak Basu
"""
import numpy as np
np.set_printoptions(threshold=np.nan)

import pickle
import sys
import time
from numpy.linalg import norm
from matplotlib import pyplot as plt

# Termination condition: when the maximum change in the
# location of the centroids is smaller than converge_dist
converge_dist = 0.1


def phi(samples, centers):
    """
    Returns the potential function value

    :ivar np.array(list of list) p: the input data points
    :ivar np.array(list of list) center: the centers
    """
    ph = 0
    for i in range(samples.shape[0]):
        ph += np.min(np.sqrt(np.sum(pow(samples[i] - centers, 2), axis=1)))
    return ph


def get_closest_center(p, centers):
    """
    Returns the closest center to p

    :ivar np.array(list) p: the input point p
    :ivar np.array(list of list) center: the centers
    """
    ind = np.argmin(np.sum(pow(p - centers, 2), axis=1))
    return ind


def D_2(p, centers):
    distance2 = np.min(np.sum(pow(p - centers, 2), axis=1))
    return distance2

def prob(samples, centers):
    total_D_2 = 0
    n = samples.shape[0]
    p = np.zeros(n)
    for i in range(n):
        p[i] = D_2(samples[i], centers)
        total_D_2 += p[i]
    p = p / total_D_2
    return p


def initialize_clusters(k):
    clusters = dict()
    for i in range(k):
        clusters[i] = []
    return clusters


def kmeans(samples, n, k):
    # Arbitrarily choose an initial k centers
    ind = np.random.randint(0, n - 1, size=k)
    C = samples[ind]  # c_0, c_1, c_{k-1}
    # print(C)
    current_dist = 100.0
    clusters = initialize_clusters(k)
    new_C = np.zeros([k, d])
    while current_dist > converge_dist:
        clusters = initialize_clusters(k)
        for i in range(n):
            closest = get_closest_center(samples[i], C)  # closest center's index
            clusters[closest].append(list(samples[i]))
        # Compute center for each cluster
        for j in range(k):
            new_C[j] = np.average(np.array(clusters[j]), axis=0)

        current_dist = np.max(np.sqrt(np.sum(pow(new_C - C, 2), axis=1)))
        C = new_C
    return C, clusters


def kmeanspp(samples, n, k):
    # Take one center c_0, chosen uniformly at random
    ind = np.random.randint(0, n - 1, size=1)
    C = samples[ind]  # c_0
    # print(C)
    for j in range(1, k):
        Pr = prob(samples, C)
        ind = np.random.choice(a=n, size=1, replace=False, p=Pr)
        C = np.row_stack((C, samples[ind]))
    # print("C=",C)
    current_dist = 100.0
    clusters = initialize_clusters(k)
    new_C = np.zeros([k, d])
    while current_dist > converge_dist:
        clusters = initialize_clusters(k)
        for i in range(n):
            closest = get_closest_center(samples[i], C)  # closest center's index
            clusters[closest].append(list(samples[i]))
        # Compute center for each cluster
        for j in range(k):
            new_C[j] = np.average(np.array(clusters[j]), axis=0)

        current_dist = np.max(np.sqrt(np.sum(pow(new_C - C, 2), axis=1)))
        C = new_C
    return C, clusters


def kmeanspp_variant(samples, n, k):
    # Take one center c_0, chosen uniformly at random
    ind = np.random.randint(0, n - 1, size=1)
    C = samples[ind]  # c_0
    # print(C)
    for j in range(1, k):
        Pr = prob(samples, C)
        ind = np.argmax(Pr)
        C = np.row_stack((C, samples[ind]))
    # print("C=",C)
    current_dist = 100.0
    clusters = initialize_clusters(k)
    new_C = np.zeros([k, d])
    while current_dist > converge_dist:
        clusters = initialize_clusters(k)
        for i in range(n):
            closest = get_closest_center(samples[i], C)  # closest center's index
            clusters[closest].append(list(samples[i]))
        # Compute center for each cluster
        for j in range(k):
            new_C[j] = np.average(np.array(clusters[j]), axis=0)

        current_dist = np.max(np.sqrt(np.sum(pow(new_C - C, 2), axis=1)))
        C = new_C
    return C, clusters


def read_data(inputfile):
    """
    Returns an n-by-d matrix, which is the dataset of n points of d dimensions

    :ivar text document
    """
    temp = inputfile.readline()
    D = []
    while temp:
        infomation = temp.strip().strip('[]')
        x = str(infomation).split(' ')
        s = [ float( x ) for x in x if x ]
        D.append(s)
        temp = inputfile.readline()
    return D


def show_clusters2D(centers, clusters, k):
    mark = ['r', 'b', 'g', 'k', 'm', 'y', 'gold', 'mediumpurple', 'maroon', 'lightseagreen',
            'cornflowerblue', 'fuchsia', 'dodgerblue', 'darkgoldenrod', 'lightcoral',
            'yellowgreen', 'darkcyan', 'orange', 'gray', 'darksage', 'cyan', 'blueviolet', 'deeppink',
            'lime', 'brown', 'peru', 'lightgray', 'sandybrown', 'yellow', 'pink',
            'steelblue', 'chocolate', 'tan', 'royalblue', 'teal', 'hotpink', 'navy', 'khaki', 'tomato',
            'seagreen', 'stateblue', 'plum', 'wheat', 'forestgreen', 'stategray']
    if k > len(mark):
        print("Your number of clusters exceeds our limit!")
        return 1
    # plot clusters
    for i in range(k):
        plt.scatter(np.array(clusters[i])[:, 0], np.array(clusters[i])[:, 1], color=mark[i], marker='o', s=4)
    # plot clusters
    for i in range(k):
        plt.plot(centers[i, 0], centers[i, 1], '.k', markersize=10)
    plt.show()
    return 0

def show_all2D(centers, clusters, k):
    mark = ['r', 'b', 'g', 'k', 'm', 'y', 'gold', 'mediumpurple', 'maroon', 'lightseagreen',
            'cornflowerblue', 'fuchsia', 'dodgerblue', 'darkgoldenrod', 'lightcoral',
            'yellowgreen', 'darkcyan', 'orange', 'gray', 'cadetblue', 'cyan', 'blueviolet', 'deeppink',
            'lime', 'brown', 'peru', 'lightgray', 'sandybrown', 'yellow', 'pink',
            'steelblue', 'chocolate', 'tan', 'royalblue', 'teal', 'hotpink', 'navy', 'khaki', 'tomato',
            'seagreen', 'stateblue', 'plum', 'wheat', 'forestgreen', 'stategray']
    if k > len(mark):
        print("Your number of clusters exceeds our limit!")
        return 1
    # plot clusters
    for i in range(k):
        plt.scatter(np.array(clusters[i])[:, 0], np.array(clusters[i])[:, 1], color=mark[i], marker='o', s=4)
    # plot clusters
    for i in range(k):
        plt.plot(centers[i, 0], centers[i, 1], '.k', markersize=10)
    # plt.show()
    return 0

if __name__ == '__main__':

    k = 25  # Number of centroids

    filename = '../Normn10000d5k25var8.txt'
    # filename = '../cloud.data'
    inf = open(filename, 'r')
    # _, filenames = read_Y_distribution(inf1)
    samples = read_data(inf)
    inf.close()

    samples = np.array(samples)
    # print(samples.shape[0])
    n = samples.shape[0]  # Number of points
    d = samples.shape[1]  # Dimension of samples
    plt.close('all')
    # fg1 = plt.figure(1)
    # plt.scatter(samples[:, 0], samples[:, 1], s=5)
    print("-------- k-means experiment: data = ", filename, "--------")
    start = time.perf_counter()
    C, clusters = kmeans(samples, n, k)
    end = time.perf_counter()
    # print("kmeans centers=\n", C)
    ph = phi(samples, C)
    print("kmeans potential function phi=%f, runtime=%f s." % (ph, end - start))
    # show_clusters2D(C, clusters, k)

    start = time.perf_counter()
    C2, clusters2 = kmeanspp(samples, n, k)
    end = time.perf_counter()
    # print("kmeans++ centers=\n", C2)
    ph2 = phi(samples, C2)
    print("kmeans++ potential function phi=%f, runtime=%f s." % (ph2, end - start))
    # show_clusters2D(C2, clusters2, k)

    start = time.perf_counter()
    C3, clusters3 = kmeanspp_variant(samples, n, k)
    end = time.perf_counter()
    # print("kmeans++ variant centers=\n", C3)
    ph3 = phi(samples, C3)
    print("kmeans++ variant potential function phi=%f, runtime=%f s." % (ph3, end - start))
    # show_clusters2D(C3, clusters3, k)

    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    show_all2D(C, clusters, k)
    plt.subplot(1, 3, 2)
    show_all2D(C2, clusters2, k)
    plt.subplot(1, 3, 3)
    show_all2D(C3, clusters3, k)
    plt.show()
