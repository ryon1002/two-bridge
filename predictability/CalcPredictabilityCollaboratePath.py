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
        assigns, costs, true_costs = [], [], []
        pre_cond = world.make_pre_cond(pre_assign)
        for order in itertools.permutations(valid_points):
            for i in range(len(valid_points)):
                assigns.append((order[:i],order[i:]))
                costs.append(world.orderd_assign_cost(assigns[-1], pre_cond))
                true_costs.append(world.orderd_assign_cost(assigns[-1], pre_cond, 1))
        return assigns, np.array(costs), np.array(true_costs)

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
            assigns, costs, true_costs = self.calc_orderd_assign_dist(world, ((), i))
            probs = self.make_prob_dist(costs, 1, 0)
            objective = probs
            # objective = costs * probs
            # print i[0], np.sum(costs * probs)
            print i[0], np.sum(true_costs * probs)
            # for j in probs.argsort()[::-1][:10]:
            #     print assigns[j], objective[j], costs[j]

    def predictable_path_action_hierarchy(self, world, t=1):
        beta = 1
        prob_goal_action = np.empty((len(world.items), len(world.action_id)))
        for i in range(len(world.items)):
            costs =np.array([world.cost_after_action(a, i) for a in range(len(world.action_id))])
            prob_goal_action[i] = self.make_prob_dist(costs, beta)
        prob_goal_action /= np.sum(prob_goal_action, axis=0)

        for a in range(len(world.action_id)):
            assigns, probs, costs = [], [], []
            for i in range(len(world.items)):
                assign, cost, _ = self.calc_orderd_assign_dist(world, ((), (i,)))
                assigns.extend(assign)
                costs.extend(cost)
                probs.extend(self.make_prob_dist(cost, beta) * prob_goal_action[i, a])
            # print a, np.dot(np.array(costs), np.array(probs))
            for j in np.array(probs).argsort()[::-1][:10]:
                print assigns[j], probs[j]

    def predictable_path_action_flat(self, world, t=1):
        beta = 1
        for a in range(len(world.action_id)):
            from scipy.spatial.distance import squareform, pdist
            world.dists = squareform(pdist(np.array(world.items + list(world.human)
                                                    + list(world.agent + world.action_id[a])),
                                           metric="cityblock"))
            assigns, costs, _ = self.calc_orderd_assign_dist(world, ((), ()))
            probs = self.make_prob_dist(costs, beta)
            print a, np.dot(np.array(costs), np.array(probs))
            for j in np.array(probs).argsort()[::-1][:10]:
                print assigns[j], probs[j], costs[j]

if __name__ == '__main__':
    from predictability import ItemPickWorld, ItemPickWorldWeight
    world = ItemPickWorld.ItemPickWorld()
    # world = ItemPickWorldWeight.ItemPickWorldWeight()
    solver = CalcPredictabilityCollaboratePath()
    # solver.predictable_path_action_hierarchy(world, 1)
    solver.predictable_path_action_flat(world, 1)
    # solver.predictable_path(world, 1)
    world.show_world()
