import itertools
import numpy as np


class CalcPredictabilityCollaboratePath(object):
    def calc_assign_dist(self, world, pre_assign):
        valid_points = [p for p in world.goal_ids if p not in set(pre_assign[0] + pre_assign[1])]
        assigns, costs = [], []
        pre_cond = world.make_pre_cond(pre_assign)
        for i in range(0, len(valid_points) + 1):
            for human_goals in itertools.combinations(valid_points, i):
                agent_goals = tuple([p for p in valid_points if p not in set(human_goals)])
                assigns.append((human_goals, agent_goals))
                costs.append(world.assign_cost((human_goals, agent_goals), pre_cond, "optimal"))
        return assigns, np.array(costs)

    def calc_orderd_assign_dist(self, world, pre_assign):
        valid_points = [p for p in world.goal_ids if p not in set(pre_assign[0] + pre_assign[1])]
        assigns, costs = [], []
        pre_cond = world.make_pre_cond(pre_assign)
        for order in itertools.permutations(valid_points):
            for i in range(len(valid_points)):
                assigns.append((order[:i],order[i:]))
                costs.append(world.orderd_assign_cost(assigns[-1], pre_cond))
        return assigns, np.array(costs)

    def make_prob_dist(self, costs, beta=1, top_filter=0):
        probs = np.exp(beta * -costs)
        if top_filter > 0:
            low_index = probs.argsort()[::-1][top_filter:]
            probs[low_index] = 0
        probs /= np.sum(probs)
        return probs

    def predictable_path(self, world, t=1):
        # for i in [(3,)]:
        for i in itertools.permutations(world.goal_ids, t):
            # assigns, costs = self.calc_assign_dist(world, ((), i))
            assigns, costs = self.calc_orderd_assign_dist(world, ((), i))
            probs = self.make_prob_dist(costs, 0.5, 2)
            objective = probs
            # objective = costs * probs
            print np.sum(costs * probs)

            print i
            for j in probs.argsort()[::-1][:10]:
                print assigns[j], objective[j], costs[j]


if __name__ == '__main__':
    from predictability import ItemPickWorld
    world = ItemPickWorld.ItemPickWorld()
    solver = CalcPredictabilityCollaboratePath()
    solver.predictable_path(world, 1)
    # world.show_world()
