import numpy as np


class ValueIteration(object):
    def get_greedy_policy(self, mdp, horizon=10000):
        q = self.value_iteration(mdp.t, mdp.r, mdp.d, horizon)
        policy = np.zeros_like(q)
        # print q[0]
        policy[q == np.max(q, axis=0)] = 1
        return policy / np.sum(policy, axis=0)

    def get_q_softmax_policy(self, mdp, beta=1, horizon=10000):
        q = self.value_iteration(mdp.t, mdp.r, mdp.d, horizon)
        policy = np.exp(q * beta)
        return policy / np.sum(policy, axis=0)

    @staticmethod
    def _value_iteration_base(t, r, d, func, horizon=10000):
        v = np.zeros(t.shape[1])
        r = np.sum(t * r, axis=2)
        td = t * d
        for _ in range(horizon):
            q = r + np.dot(td, v)
            n_v = func(q, axis=0)
            if np.linalg.norm(n_v - v, np.inf) < 1e-12:
                break
            v = n_v
        return r + np.dot(td, v), v

    def value_iteration(self, t, r, d, horizon=10000):
        q, _ = self._value_iteration_base(t, r, d, np.max, horizon)
        return q

    @staticmethod
    def softmax(arr, axis):
        arr_max = np.max(arr, axis=axis)
        return arr_max + np.log(np.sum(np.exp(arr - arr_max), axis=axis))
