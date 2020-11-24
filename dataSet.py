import numpy as np
import matplotlib.pyplot as plt
# import sklearn
from sklearn import datasets


def dbmoon(N=100, d=2, r=10, w=2):
    N1 = 10 * N
    w2 = w / 2
    done = True
    data = np.empty(0)
    while done:
        # generate Rectangular data
        tmp_x = 2 * (r + w2) * (np.random.random([N1, 1]) - 0.5)
        tmp_y = (r + w2) * np.random.random([N1, 1])
        tmp = np.concatenate((tmp_x, tmp_y), axis=1)
        tmp_ds = np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)
        # generate double moon data ---upper
        idx = np.logical_and(tmp_ds > (r - w2), tmp_ds < (r + w2))
        idx = (idx.nonzero())[0]

        if data.shape[0] == 0:
            data = tmp.take(idx, axis=0)
        else:
            data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
        if data.shape[0] >= N:
            done = False
    print(data,data.shape)
    db_moon = data[0:N, :]
    print(db_moon,db_moon.shape)
    # generate double moon data ----down
    data_t = np.empty([N, 2])
    data_t[:, 0] = data[0:N, 0] + r
    data_t[:, 1] = -data[0:N, 1] - d
    db_moon = np.concatenate((db_moon, data_t), axis=0)
    print(db_moon.shape)
    return db_moon

if __name__ == '__main__':
    N = 200
    d = -2
    r = 10
    w = 2
    a = 0.1
    num_MSE = []
    num_step = []
    data = dbmoon(N, d, r, w)

    # plt.plot(data[0:N, 0], data[0:N, 1], 'r*', data[N:2 * N, 0], data[N:2 * N, 1], 'b*')
    # plt.plot()
    # plt.show()
    X, y = datasets.make_moons(n_samples=100, n_features=10,
                                            noise=0.1)
    plt.scatter(X[:, 0], y)
    plt.show()