import numpy as np
import itertools


class MOMDP(object):
    def __init__(self, x, y, a):
        self.x = x
        self.y = y
        self.a = a
        self.tx = np.zeros((self.y, self.a, self.x, self.x))
        self.ty = np.zeros((self.y, self.a, self.x, self.y))
        self.r = np.zeros((self.y, self.a, self.x, self.x))

    def pre_culc(self):
        # For calc Avector
        self.valid_nx_x = np.sum(self.tx, axis=0) > 0
        self.valid_nx_x = {a: {x: np.where(self.valid_nx_x[a][x])[0] for x in range(self.valid_nx_x.shape[1])}
                           for a in range(self.valid_nx_x.shape[0])}
        self.valid_nx_y = {a: {x: np.sum(self.ty[:, a, x, :], axis=0) for x in range(self.x)} for a in range(self.a)}

    def calc_a_vector(self, d=1, bs=None, with_a=True):
        if d == 1:
            self.aVector = {x: self.r[:, x, :].copy() for x in range(self.x)}
            return
        self.calc_a_vector(d - 1, bs, False)
        print d
        a_vector = {}
        for x in range(self.x):
            a_vector[x] = {}
            for a in range(self.a):
                nx_x = self.tx[:, a, x]
                ai_list = [nx_x[:, nx] * self.aVector[nx] * self.valid_nx_y[a][x] for nx in self.valid_nx_x[a][x]]
                ai_len_list = [len(ai_list[i]) for i in range(len(ai_list))]

                ai_list2 = np.zeros((np.prod(ai_len_list), self.y))
                for m, i in enumerate(itertools.product(*[range(l) for l in ai_len_list])):
                    ai_list2[m] = np.sum([ai_list[n][j] for n, j in enumerate(i)], axis=0)
                ai_list2 = self.unique_for_raw(ai_list2)
                a_vector[x][a] = self.r[a, x] + ai_list2
        if with_a:
            self.a_vector_a = {x: {a: self.prune(vector, bs) for a, vector in vectorA.viewitems()} for x, vectorA in
                               a_vector.viewitems()} if bs is not None else a_vector
        else:
            self.aVector = {x: self.prune(np.concatenate(vector.values(), axis=0), bs) for x, vector in
                            a_vector.viewitems()} if bs is not None else a_vector

    @staticmethod
    def unique_for_raw(a):
        return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))) \
            .view(a.dtype).reshape(-1, a.shape[1])

    @staticmethod
    def prune(a_vector, bs):
        index = np.unique(np.argmax(np.dot(a_vector, bs.T), axis=0))
        return a_vector[index]

    def value_a(self, x, a, b):
        return np.max(np.dot(self.a_vector_a[x][a], b))

    def get_best_action(self, x, b):
        value_map = {k: np.max(np.dot(v, b)) for k, v in self.a_vector_a[x].viewitems()}
        return sorted(value_map.viewitems(), key=lambda a: a[1])[-1][0]
