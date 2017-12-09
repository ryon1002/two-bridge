import numpy as np


class Agent(object):
    # def __init__(self, world):
    #     self.world = world
    # def init_state(self):
    #     return 0

    def calc_policy(self, select_rule, horizon=10000, **kwargs):
        if select_rule == "greedy":
            q = self.value_iteration(self.world.t, self.world.r, self.world.d, horizon)
            self.policy = np.zeros_like(q)
            self.policy[q == np.max(q, axis=0)] = 1
            self.policy /= np.sum(self.policy, axis=0)
        elif select_rule == "max_q_softmax":
            q = self.value_iteration(self.world.t, self.world.r, self.world.d, horizon)
            self.q = q
            self.policy = np.exp(q * kwargs["beta"])
            self.policy /= np.sum(self.policy, axis=0)
        # elif selectRule == "greedy_argmax":
        #     q = self.valueItaration(self.world.t, self.world.r, self.world.d)
        #     self.policy = np.zeros_like(q)
        #     self.policy[np.argmax(q, axis=0), np.arange(self.world.s)] = 1
        # elif selectRule == "greedy_with_invalid_greedy":
        #     q = self.valueItarationwithInvalid(self.world.t, self.world.r, self.world.d)
        #     self.policy = np.zeros_like(q)
        #     self.policy[np.argmax(q, axis=0), np.arange(self.world.s)] = 1
        elif select_rule == "soft_greedy":
            self.logPolicy = self.soft_value_iteration(self.world.t, self.world.r, self.world.d)
            self.policy = np.exp(self.logPolicy)
        # elif selectRule == "policy_iteration":
        #     if "initPolicy" not in keywords:
        #         raise ValueError("Missing initPolicy")
        #     self.policyIteration(self.world.t, self.world.r, self.world.d, keywords["initPolicy"])
        else:
            raise ValueError("No implemented method.")

            # def policyIteration(self, t, r, d, init_pi, max_iter=50):
            #     self.policy = init_pi
            #     for _i in range(max_iter):
            #         q, v = self.policyEvaliate(t, r, d)
            #         nextPolicy = np.zeros_like(q)
            #         nextPolicy[np.argmax(q, axis=0), np.arange(self.world.s)] = 1
            #         if np.linalg.norm(nextPolicy - self.policy, np.inf) < 1e-4:
            #             break
            #         self.policy = nextPolicy
            #
            #     self.policy = nextPolicy
            #     self.q = q
            #
            # def policyEvaliate(self, t, r, d):
            #     q, v = self._valueIterationBase(t, r, d, self.calcExpectedValueUnderPolicy)
            #     return q, v
            #

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
        q, v = self._value_iteration_base(t, r, d, np.max, horizon)
        return q

        # def valueItarationwithInvalid(self, t, r, d):
        #     q, v = self._valueIterationBase(t, r, d, self.maxwithInvalid)
        #     return q
        #

    def soft_value_iteration(self, t, r, d):
        q, v = self._value_iteration_base(t, r, d, self.softmax)
        return q - v

        # def maxwithInvalid(self, arr, axis):
        #     arr *= self.world.validSA
        #     return np.max(arr, axis=axis)
        #
        # def calcExpectedValueUnderPolicy(self, arr, axis):
        #     arr *= self.policy
        #     return np.sum(arr, axis=axis)
        #

    @staticmethod
    def softmax(arr, axis):
        arr_max = np.max(arr, axis=axis)
        return arr_max + np.log(np.sum(np.exp(arr - arr_max), axis=axis))

    # def move(self, start_state=None, horizon=100):
    #     state = start_state if start_state is not None else self.init_state()
    #     states = [state]
    #     actions = []
    #     for _i in range(horizon):
    #         action = np.random.choice(np.arange(self.world.a), p=self.policy[:, state])
    #         state = self.world.get_next_state(action, state)
    #         actions.append(action)
    #         states.append(state)
    #         if self.world.is_terminate(state):
    #             break
    #     return states, actions

    def move(self, mdp, policy, start_state, horizon=100):
        state = start_state
        states = [state]
        actions = []
        for _i in range(horizon):
            action = np.random.choice(np.arange(mdp.a), p=policy[:, state])
            state = mdp.get_next_state(action, state)
            actions.append(action)
            states.append(state)
            if mdp.is_terminate(state):
                break
        return states, actions

    def calc_policy_and_move(self, select_rule, start_state=None, horizon=100):
        self.calc_policy(select_rule)
        return self.move(start_state, horizon)

    def do_actions(self, actions, start_state=None):
        state = start_state if start_state is not None else self.init_state()
        states = [state]
        for a in actions:
            states.append(self.world.get_next_state(a, states[-1]))
        return states
