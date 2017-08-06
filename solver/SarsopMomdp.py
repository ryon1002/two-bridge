import numpy as np
import os
import itertools
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict


class SarsopMOMDP(object):
    def solve(self, momdp, start_state, delete_mid_file=True):
        input_file = "solver/ext/momdp.pomdpx"
        output_file = "solver/ext/momdp.policy"
        pomdpx = self.make_pomdpx(momdp, start_state)
        with open(input_file, "w") as f:
            f.write(self.prettify(pomdpx))
        subprocess.call("\"solver/ext/pomdpsol.exe\" " + input_file + " -o " + output_file)
        policy = self.load_policy(output_file)
        if delete_mid_file:
            os.remove(input_file)
            os.remove(output_file)
        return policy

    @staticmethod
    def make_pomdpx(momdp, start_state):
        pomdpx = ET.Element("pomdpx", version="0.1", id="autogeterated")

        discount = ET.SubElement(pomdpx, "Discount")
        discount.text = "0.99"

        variable = ET.SubElement(pomdpx, "Variable")
        state = ET.SubElement(variable, "StateVar", vnamePrev="state_0", vnameCurr="state_1", fullyObs="true")
        v_state = ET.SubElement(state, "NumValues")
        v_state.text = str(momdp.x)
        intention = ET.SubElement(variable, "StateVar", vnamePrev="intention_0", vnameCurr="intention_1")
        v_intention = ET.SubElement(intention, "ValueEnum")
        v_intention.text = " ".join(["i" + str(y) for y in range(momdp.y)])
        observation = ET.SubElement(variable, "ObsVar", vname="observation")
        v_observation = ET.SubElement(observation, "ValueEnum")
        v_observation.text = "obs"
        action = ET.SubElement(variable, "ActionVar", vname="action")
        v_action = ET.SubElement(action, "ValueEnum")
        v_action.text = " ".join(["a" + str(y) for y in range(momdp.a)])
        _reward = ET.SubElement(variable, "RewardVar", vname="reward")

        initial_state_belief = ET.SubElement(pomdpx, "InitialStateBelief")
        initial_state_state_prob = ET.SubElement(initial_state_belief, "CondProb")
        initial_state_state_var = ET.SubElement(initial_state_state_prob, "Var")
        initial_state_state_var.text = "state_0"
        initial_state_state_parent = ET.SubElement(initial_state_state_prob, "Parent")
        initial_state_state_parent.text = "null"
        initial_state_state_param = ET.SubElement(initial_state_state_prob, "Parameter", type="TBL")
        initial_state_state_entry = ET.SubElement(initial_state_state_param, "Entry")
        initial_state_state_instance = ET.SubElement(initial_state_state_entry, "Instance")
        initial_state_state_instance.text = "-"
        initial_state_state_prob_table = ET.SubElement(initial_state_state_entry, "ProbTable")
        init_state = [0] * momdp.x
        init_state[start_state] = 1
        initial_state_state_prob_table.text = " ".join([str(i) for i in init_state])
        initial_state_intention_prob = ET.SubElement(initial_state_belief, "CondProb")
        initial_state_intention_var = ET.SubElement(initial_state_intention_prob, "Var")
        initial_state_intention_var.text = "intention_0"
        initial_state_intention_parent = ET.SubElement(initial_state_intention_prob, "Parent")
        initial_state_intention_parent.text = "null"
        initial_state_intention_param = ET.SubElement(initial_state_intention_prob, "Parameter", type="TBL")
        initial_state_intention_entry = ET.SubElement(initial_state_intention_param, "Entry")
        initial_state_intention_instance = ET.SubElement(initial_state_intention_entry, "Instance")
        initial_state_intention_instance.text = "-"
        initial_state_intention_prob_table = ET.SubElement(initial_state_intention_entry, "ProbTable")
        initial_state_intention_prob_table.text = "uniform"

        transition_function = ET.SubElement(pomdpx, "StateTransitionFunction")
        transition_state_prob = ET.SubElement(transition_function, "CondProb")
        transition_state_var = ET.SubElement(transition_state_prob, "Var")
        transition_state_var.text = "state_1"
        transition_state_parent = ET.SubElement(transition_state_prob, "Parent")
        transition_state_parent.text = "action state_0 intention_0"
        transition_state_param = ET.SubElement(transition_state_prob, "Parameter", type="TBL")
        for i, a, x in itertools.product(range(momdp.tx.shape[0]), range(momdp.tx.shape[1]), range(momdp.tx.shape[2])):
            for n_x in np.where(momdp.tx[i, a, x] > 0)[0]:
                transition_state_entry = ET.SubElement(transition_state_param, "Entry")
                transition_state_instance = ET.SubElement(transition_state_entry, "Instance")
                transition_state_instance.text = "a" + str(a) + " s" + str(x) + " i" + str(i) + " s" + str(n_x)
                transition_state_prob_table = ET.SubElement(transition_state_entry, "ProbTable")
                transition_state_prob_table.text = str(momdp.tx[i, a, x, n_x])
        transition_intention_prob = ET.SubElement(transition_function, "CondProb")
        transition_intention_var = ET.SubElement(transition_intention_prob, "Var")
        transition_intention_var.text = "intention_1"
        transition_intention_parent = ET.SubElement(transition_intention_prob, "Parent")
        transition_intention_parent.text = "action intention_0"
        transition_intention_param = ET.SubElement(transition_intention_prob, "Parameter", type="TBL")
        transition_intention_entry = ET.SubElement(transition_intention_param, "Entry")
        transition_intention_instance = ET.SubElement(transition_intention_entry, "Instance")
        transition_intention_instance.text = "* - -"
        transition_intention_prob_table = ET.SubElement(transition_intention_entry, "ProbTable")
        transition_intention_prob_table.text = "1.0 0.0 0.0 1.0"

        obs_function = ET.SubElement(pomdpx, "ObsFunction")
        obs_function_prob = ET.SubElement(obs_function, "CondProb")
        obs_function_var = ET.SubElement(obs_function_prob, "Var")
        obs_function_var.text = "observation"
        obs_function_parent = ET.SubElement(obs_function_prob, "Parent")
        obs_function_parent.text = "action state_1 intention_1"
        obs_function_param = ET.SubElement(obs_function_prob, "Parameter", type="TBL")
        obs_function_entry = ET.SubElement(obs_function_param, "Entry")
        obs_function_instance = ET.SubElement(obs_function_entry, "Instance")
        obs_function_instance.text = "* * * -"
        obs_function_prob_table = ET.SubElement(obs_function_entry, "ProbTable")
        obs_function_prob_table.text = "1.0"

        reward_function = ET.SubElement(pomdpx, "RewardFunction")
        reward_function_func = ET.SubElement(reward_function, "Func")
        reward_function_var = ET.SubElement(reward_function_func, "Var")
        reward_function_var.text = "reward"
        reward_function_parent = ET.SubElement(reward_function_func, "Parent")
        reward_function_parent.text = "action state_0 intention_0"
        reward_function_param = ET.SubElement(reward_function_func, "Parameter", type="TBL")
        for a, i in itertools.product(range(momdp.r.shape[0]), range(momdp.r.shape[2])):
            reward_function_entry = ET.SubElement(reward_function_param, "Entry")
            reward_function_instance = ET.SubElement(reward_function_entry, "Instance")
            reward_function_instance.text = "a" + str(a) + " - i" + str(i)
            reward_function_prob_table = ET.SubElement(reward_function_entry, "ValueTable")
            reward_function_prob_table.text = " ".join([str(p) for p in momdp.r[a, :, i]])
        return pomdpx

    def load_policy(self, policy_file):
        alpha_vector = ET.parse(policy_file).getroot().find("AlphaVector")
        a_vector_a = defaultdict(lambda: defaultdict(lambda: np.zeros((0, int(alpha_vector.attrib["vectorLength"])))))
        for vector in alpha_vector.findall("Vector"):
            x = int(vector.attrib["obsValue"])
            a = int(vector.attrib["action"])
            v = np.array([[float(i) for i in vector.text.strip().split(" ")]])
            a_vector_a[x][a] = np.concatenate((a_vector_a[x][a], v), axis=0)
        return {k: {k2: v2 for k2, v2 in v.viewitems()} for k, v in a_vector_a.viewitems()}

    @staticmethod
    def prettify(elem):
        rough_string = ET.tostring(elem, 'utf-8')
        parse_data = minidom.parseString(rough_string)
        return parse_data.toprettyxml(indent="  ")
