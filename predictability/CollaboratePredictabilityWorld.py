import itertools
class CollaboratePredictabilityWorld(object):
    def make_pre_cond(self, pre_assign):
        raise NotImplementedError

    def single_optimal_assign_cost(self, assign, pre_cond):
        return min([self.single_orderd_assign_cost(orderd_assign, pre_cond)
                    for orderd_assign in itertools.permutations(assign) ])

    def assign_cost(self, o_assign, pre_cond, cost_type="optimal"):
        if cost_type == "optimal":
            cost_func = self.single_optimal_assign_cost
        return max(cost_func(o_assign[0], pre_cond[0]), cost_func(o_assign[1], pre_cond[1]))

    def orderd_assign_cost(self, o_assign, pre_cond):
        return max(self.single_orderd_assign_cost(o_assign[0], pre_cond[0]),
                    self.single_orderd_assign_cost(o_assign[1], pre_cond[1]))

    def single_orderd_assign_cost(self, o_assign, pre_cond):
        raise NotImplementedError
