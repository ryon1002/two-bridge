import numpy as np
import itertools


class Tiger(object):
    def __init__(self, depth=1, bs=None):
        #         self.x = 6
        self.x = 6
        self.y = 2
        self.a = 3
        self.t = np.zeros((self.y, self.a, self.x, self.x))
        self.r = np.zeros((self.a, self.x, self.y))

        for x in range(self.x - 1):
            self.t[0, -1, x, min(x + 1, self.x - 2)] = 0.85
            self.t[0, -1, x, max(0, x - 1)] = 0.15
            self.t[1, -1, x, min(x + 1, self.x - 2)] = 0.15
            self.t[1, -1, x, max(0, x - 1)] = 0.85
        self.t[0, :-1, :, -1] = 1
        self.t[1, :-1, :, -1] = 1
        self.t[:, :, -1, -1] = 1

        self.r[0, :-1, 0] = 10
        self.r[1, :-1, 0] = -100
        self.r[0, :-1, 1] = -100
        self.r[1, :-1, 1] = 10
        self.r[2, :-1, :] = -1

        self.calc_a_vector(depth, bs)

    def calc_a_vector(self, d=1, bs=None):
        if d == 1:
            self.a_vector = {x: self.r[:, x, :].copy() for x in range(self.x)}
            return
        self.calc_a_vector(d - 1, bs)
        a_vector = {}
        for x in range(self.x):
            a_vector[x] = np.zeros((0, self.y))
            for a in range(self.a):
                ai_list = []
                ai_len_list = []
                nxs = self.t[:, a, x]
                valid_nxs = np.where(np.sum(nxs, axis=0) > 0)[0]
                for nx in valid_nxs:
                    ai_list.append(nxs[:, nx] * self.a_vector[nx])
                    ai_len_list.append(range(len(self.a_vector[nx])))

                ai_list2 = np.zeros((0, self.y))
                for i in itertools.product(*ai_len_list):
                    ai_list2 = np.vstack((ai_list2, np.sum([ai_list[n][j] for n, j in enumerate(i)], axis=0)))
                ai_list2 = self.unique_for_raw(ai_list2)
                a_vector[x] = np.vstack((a_vector[x], self.r[a, x] + ai_list2))
        self.a_vector = {x: self.prune(vector, bs) for x, vector in a_vector.viewitems()} if bs is not None else a_vector

    @staticmethod
    def unique_for_raw(a):
        return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))).view(a.dtype).reshape(-1,
                                                                                                           a.shape[1])

    def value(self, x, b):
        return np.max(np.dot(self.a_vector[x], b))

    @staticmethod
    def prune(a_vector, bs):
        index = np.unique(np.argmax(np.dot(a_vector, bs.T), axis=0))
        return a_vector[index]


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     b1 = np.arange(0, 1.01, 0.04)
#     b2 = 1 - b1
#
#     b = np.concatenate(([b1], [b2]), axis=0).T
#
#     env = Tiger(1, b)
#     v = np.array([env.value(1, b[i]) for i in range(len(b))])
#     plt.plot(b[:, 0], v)
#
#     env2 = Tiger(10, b)
#     v = np.array([env2.value(1, b[i]) for i in range(len(b))])
#     plt.plot(b[:, 0], v)
#
#     env3 = Tiger(20, b)
#     v = np.array([env3.value(1, b[i]) for i in range(len(b))])
#     plt.plot(b[:, 0], v)
#     #
#     #     import datetime
#     #     start = datetime.datetime.now()
#     #     env4 = Tiger(30, b)
#     #     v = np.array([env4.value(b[i]) for i in range(len(b))])
#     #     print datetime.datetime.now() - start
#     #     plt.plot(b[:, 0], v)
#
#     plt.show()
