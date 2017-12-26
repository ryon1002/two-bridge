import numpy as np
from predictability import ItemPickWorldProperty


class CalcPredictabilityCollaboratePath(object):
    # def calc_assign_dist(self, world, pre_assign):
    #     valid_points = [p for p in world.goal_ids if p not in set(pre_assign[0] + pre_assign[1])]
    #     assigns, costs = [], []
    #     pre_cond = world.make_pre_cond(pre_assign)
    #     for i in range(0, len(valid_points) + 1):
    #         for human_goals in itertools.combinations(valid_points, i):
    #             agent_goals = tuple([p for p in valid_points if p not in set(human_goals)])
    #             assigns.append((human_goals, agent_goals))
    #             costs.append(world.assign_cost((human_goals, agent_goals), pre_cond, "optimal"))
    #     return assigns, np.array(costs)

    @staticmethod
    def make_prob_dist(objective, beta=1, top_filter=0):
        prob = np.exp(beta * objective)
        if top_filter > 0:
            low_index = prob.argsort()[::-1][top_filter:]
            prob[low_index] = 0
        prob /= np.sum(prob)
        return prob

    @staticmethod
    def action_reward_dist(world, conditions, reward=None):
        actions = [a for a in world.action.get_all_action_seq(conditions)]
        reward = np.array([world.get_reward(a, reward) for a in actions])
        return actions, reward

    def predictable_path(self, world, t=1):
        for condition in world.action.get_all_condition(t):
            actions, reward = self.action_reward_dist(world, condition)
            prob = self.make_prob_dist(reward, 1, 0)
            print condition, np.sum(-reward * prob)
            # print i[0], np.sum(true_costs * probs)
            # for j in probs.argsort()[::-1][:10]:
            #     print assigns[j], objective[j], costs[j]

    def reward_prob_dist(self, world, rewards, conditions):
        reward_list = np.array([world.get_reward()])

    # def predictable_path_action_hierarchy(self, world, t=1):
    #     beta = 1
    #     prob_goal_action = np.empty((len(world.items), len(world.action_id)))
    #     for i in range(len(world.items)):
    #         costs = np.array([world.cost_after_action(a, i) for a in range(len(world.action_id))])
    #         prob_goal_action[i] = self.make_prob_dist(costs, beta)
    #     prob_goal_action /= np.sum(prob_goal_action, axis=0)
    #     for a in range(len(world.action_id)):
    #         assigns, probs, costs = [], [], []
    #         for i in range(len(world.items)):
    #             assign, cost, _ = self.calc_ordered_assign_dist(world, ((), (i,)))
    #             assigns.extend(assign)
    #             costs.extend(cost)
    #             probs.extend(self.make_prob_dist(cost, beta) * prob_goal_action[i, a])
    #         # print a, np.dot(np.array(costs), np.array(probs))
    #         for j in np.array(probs).argsort()[::-1][:10]:
    #             print assigns[j], probs[j]

    # def calc_prob_reward_given_subgoal(self, world, rewards):
    #     prob_list = np.empty((len(world.items), len(rewards)))
    #     for r_i, reward in enumerate(rewards):
    #         # world.set_cost(reward)
    #         human_reward = {0: 0, 1: 0, 2: 0}
    #         assigns, costs, true_cost = self.calc_ordered_assign_dist(world, ((), ()), human_reward,
    #                                                                   reward)
    #         probs = self.make_prob_dist(costs, 1)
    #         for target in range(6):
    #             target_assigns = [i for i, a in enumerate(assigns) if
    #                               len(a[1]) > 0 and a[1][0] == target]
    #             prob_list[target][r_i] = np.sum(probs[target_assigns])
    #     prob_list /= np.sum(prob_list, axis=1, keepdims=True)
    #     prob_list /= np.sum(prob_list, axis=0, keepdims=True)
    #     return prob_list
    #
    # def calc_prob_subgoal_gven_action(self, world):
    #     beta = 1
    #     prob_goal_action = np.empty((len(world.items), len(world.action_id)))
    #     for i in range(len(world.items)):
    #         costs = np.array([world.cost_after_action(a, i) for a in range(len(world.action_id))])
    #         print costs
    #         prob_goal_action[i] = self.make_prob_dist(costs, beta)
    #     prob_goal_action /= np.sum(prob_goal_action, axis=0)
    #     return prob_goal_action
    #
    # def calc_prob_reward_given_action(self, world):
    #     reward_list = [{0: 0, 1: 0, 2: 0}, {0: 0, 1: 5, 2: 0}, {0: 0, 1: 0, 2: 5}]
    #     p_sa = self.calc_prob_subgoal_gven_action(world)
    #     p_rs = self.calc_prob_reward_given_subgoal(world, reward_list)
    #     p_ra = np.dot(p_sa.T, p_rs)
    #
    # def calc_colab(self, world):
    #     reward_id = 2
    #     for i in range(6):
    #         reward_list = [{0: 0, 1: 0, 2: 0}, {0: 0, 1: 5, 2: 0}, {0: 0, 1: 0, 2: 5}]
    #         reward_prob = self.calc_prob_reward_given_subgoal(world, reward_list)
    #         score = 0
    #         for r in range(len(reward_list)):
    #             assign, cost, _ = self.calc_ordered_assign_dist(world, ((), (i,)), reward_list[r],
    #                                                             reward_list[reward_id])
    #             # assign, cost, _ = self.calc_orderd_assign_dist(world, ((), (i,)), reward_list[r], reward_list[])
    #             probs = self.make_prob_dist(cost, 1, 0)
    #             score += np.sum(cost * probs) * reward_prob[i, r]
    #         print score
    #         # print reward_prob[reward_id]


if __name__ == '__main__':
    # sample_world = ItemPickWorld.ItemPickWorld()
    # sample_world = ItemPickWorldProperty.ItemPickWorldProperty({0:0, 1:0, 2:5})
    sample_world = ItemPickWorldProperty.ItemPickWorldProperty()
    # world = ItemPickWorldWeight.ItemPickWorldWeight()
    solver = CalcPredictabilityCollaboratePath()
    # solver.calc_prob_reward_given_subgoal(world)
    # solver.calc_prob_reward_given_action(world)
    # solver.calc_colab(world)
    # solver.predictable_path_action_hierarchy(world, 1)
    # solver.predictable_path_action_flat(world, 1)
    sample_world.set_reward([{0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}])
    reward_list = [({0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}),
                   ({0: 0, 1: 0, 2: 0}, {0: 0, 1: 5, 2: 0}),
                   ({0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 5})]

    solver.predictable_path(sample_world, 1)
    # world.show_world()
