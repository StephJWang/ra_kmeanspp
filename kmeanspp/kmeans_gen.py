"""
Authors: Jun Wang
         Farhad Mohsin
         Shaunak Basu
"""
import numpy as np
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)


if __name__ == '__main__':
    print("--------k-means----------")
    n = 10000  # Number of points
    d = 5  # Dimension of samples
    k = 25  # Number of centroids
    nC = 25  # Number of “real” centers
    C = np.random.uniform(0, 500, size=[nC, d])
    # print(C)
    # print(C[1],C[0][1])
    samples = np.random.normal(C[0], 8, size=[int(n/k), d])
    for i in range(1, nC):
        s = np.random.normal(C[i], 8, size=[int(n/k), d])
        samples = np.row_stack((samples, s))

    result = open('../Normn10000d5k25var8.txt', 'w+')
    result.write(str(samples))
    result.close()
