# Problem Statement 1

import seaborn as sns
import numpy as np
from scipy.stats import poisson

obstacles = [(1,1)]
goal_states = ((1,3),(2,3))

# # non - deterministic action (equally probable)
# action_probability = {'L':0.25,'R':0.25,'U':0.25,'D':0.25}

# environment action corresponding to Agent if it does not follow the desired direction (i.e follow perpendicular direction to desired one)
env_left = {'L':'D','R':'U','U':'L','D':'R'}
env_right = {'L':'U','R':'D','U':'R','D':'L'}

#check validity of the cell
def is_valid(i,j):
    return (i,j) not in obstacles and i >= 0 and i < 3 and j >= 0 and j < 4

#print matrix after convergence 
def print_values_matrix(V):
  for i in range(2,-1,-1):
    print(" ")
    for j in range(4):
      v = V[i][j]
      print(" %.2f|" % v, end="")
    print("")

#take action
def get_next_state(action,i,j):
    if action == 'L':
        return (i,j-1)
    elif action == 'R':
        return (i,j+1)
    elif action == 'U':
        return (i+1,j)
    elif action == 'D':
        return (i-1,j)   
    else:
        return (-1,-1)

def calculate_value_function(i,j,reward,reward_matrix,discount_factor=1):
    value = 0
    for action in ['L','R','U','D']:
        # desired action with 0.8 probability
        state_x,state_y = get_next_state(action,i,j)
        if is_valid(state_x,state_y):
            desired_action_value = (reward_matrix[state_x][state_y] + discount_factor*V_pie[state_x][state_y])
        else:
            desired_action_value = (reward_matrix[i][j] + discount_factor*V_pie[i][j])
        
        # environment action with 0.1 probability
        state_x,state_y = get_next_state(env_left[action],i,j)
        if is_valid(state_x,state_y):
            env_action_left_value = (reward_matrix[state_x][state_y] + discount_factor*V_pie[state_x][state_y])
        else:
            env_action_left_value = (reward_matrix[i][j] + discount_factor*V_pie[i][j])
        
        # environment action with 0.1 probability 
        state_x,state_y = get_next_state(env_right[action],i,j)
        if is_valid(state_x,state_y):
            env_action_right_value = (reward_matrix[state_x][state_y] + discount_factor*V_pie[state_x][state_y])
        else:
            env_action_right_value = (reward_matrix[i][j] + discount_factor*V_pie[i][j])
        
        value_to_action = desired_action_value*0.8+env_action_left_value*0.1+env_action_right_value*0.1        

        value += value_to_action*0.25 # # non - deterministic action (equally probable)

    return value

# iterative policy evaluation
def iterative_policy_evaluation(iter,epsilon,reward,reward_matrix,V_pie):
    while True:
        delta = 0
        for i in range(3):
            for j in range(4):
                state = (i,j)
                if state in goal_states or state in obstacles:  # continue if encounter terminal state or obstacles
                    continue
                v = V_pie[i][j]
                V_pie[i][j] = calculate_value_function(i,j,reward,reward_matrix)
                delta = max(delta,abs(v-V_pie[i][j]))
        iter += 1
        if delta < epsilon:
            print(f"Number of iterations to converge = {iter}")
            break 
    print_values_matrix(V_pie)

# initialize the reward matrix with given reward value except the terminal states
def update_reward_matrix(reward):
    reward_matrix = [[reward for _ in range(4)] for _ in range(3)]
    reward_matrix[2][3] = 1
    reward_matrix[1][3] = -1
    return reward_matrix

# initialize V_pie with all zeroes at start
def initialize_V_pie():
    V_pie = [[0 for _ in range(4)]for _ in range(3)]
    return V_pie
rewards = [-0.04,-2,0.1,0.02,1]
epsilon = 1e-8
print("Value Functions corresponding to optimal policy\n")
for reward in rewards:
    print(f"For r(S) : {reward}")
    reward_matrix = update_reward_matrix(reward)
    V_pie = initialize_V_pie()
    iterative_policy_evaluation(0,epsilon,reward,reward_matrix,V_pie)
    print("\n")

class PoissonCalculator(object):
    pmf_c = {}
    sf_c = {}
    ch = {}
    max_coff = 30

    @classmethod
    def pmf_get(cls, mu, cutoff):
        assert isinstance(mu, int), "Parameter 'mu' should be an integer."
        assert isinstance(cutoff, int), "Parameter 'cutoff' should be an integer."

        if (mu, cutoff) not in cls.ch:
            cls.pmf_cal_series(mu, cutoff)

        return cls.ch[(mu, cutoff)]

    @classmethod
    def pmf_cal_series(cls, mu, cutoff):
        if mu not in cls.pmf_c:
            print("Calculating Poisson distribution...")

            pmf = np.zeros(cls.max_coff + 1)
            sf = np.zeros(cls.max_coff + 1)
            for i in range(cls.max_coff + 1):
                pmf[i] = np.exp(-mu) * (mu ** i) / np.math.factorial(i)
                sf[i] = 1 - np.sum(pmf[:i+1])

            cls.pmf_c[mu] = pmf
            cls.sf_c[mu] = sf

        out = np.copy(cls.pmf_c[mu][:cutoff+1])
        out[-1] += cls.sf_c[mu][cutoff]

        cls.ch[(mu, cutoff)] = out

class PolicyIS(object):

    capacity = 20
    rental_reward = 10.
    cost_of_moving = 2.
    max_movement_allowed = 5
    parking_cost = 4.

    bad_action_effect = 100.

    request_mean_G1 = 3
    request_mean_G2 = 4
    return_mean_G1 = 3
    return_mean_G2 = 2

    discount = 0.9

    PolicyEvaluationError = 0.01

    policy = None
    value = None

    def _init_(self):
        self.policy = np.zeros([self.capacity + 1]*2, int)
        self.value = np.zeros([self.capacity + 1]*2)

        self._reward1 = self.expected_rental_reward(self.request_mean_G1)
        self._reward2 = self.expected_rental_reward(self.request_mean_G2)

        assert self.bad_action_effect >= 0

    def bellman(self, action, s1, s2):
        transp1 = self.transition_probabilty(s1, self.request_mean_G1, self.return_mean_G1, -action)
        transp2 = self.transition_probabilty(s2, self.request_mean_G2, self.return_mean_G2, action)
        transp = np.outer(transp1, transp2)

        return self._reward1[s1] + self._reward2[s2] - self.expected_cost_of_moving(s1, s2, action) + \
               self.discount * sum((transp * self.value).flat)

    # policy evaluation
    def policy_evaluation(self):
        while True:
            diff = 0.
            it = np.nditer([self.policy], flags=['multi_index'])

            while not it.finished:
                action = it[0]
                s1, s2 = it.multi_index

                _temp = self.value[s1, s2]

                self.value[s1, s2] = self.bellman(action=action, s1=s1, s2=s2)

                diff = max(diff, abs(self.value[s1, s2] - _temp))

                it.iternext()

            print(diff)
            if diff < self.PolicyEvaluationError:
                break

    def policy_update(self):
        is_policy_changed = False

        it = np.nditer([self.policy], flags=['multi_index'])
        while not it.finished:
            s1, s2 = it.multi_index

            _max_val = -1
            _pol = None

            for act in range(-self.max_movement_allowed, self.max_movement_allowed + 1):
                _val = self.bellman(action=act, s1=s1, s2=s2)
                if _val > _max_val:
                    _max_val = _val
                    _pol = act

            if self.policy[s1, s2] != _pol:
                is_policy_changed = True
                self.policy[s1, s2] = _pol

            it.iternext()

        return is_policy_changed

    def expected_cost_of_moving(self, s1, s2, action):
        if action == 0:
            return 0.

        # moving from state s1 into state s2
        if action > 0:
            p = self.transition_probabilty(s1, self.request_mean_G1, self.return_mean_G1)
            cost = self._gen_move_cost_array(action)
            if action > 10:
              cost += self.parking_cost * (action - 10)
            return cost.dot(p)

        # moving from state s2 into state s1
        p = self.transition_probabilty(s2, self.request_mean_G2, self.return_mean_G2)
        cost = self._gen_move_cost_array(action)
        if action > 10:
            cost += self.parking_cost * (action - 10)
        return cost.dot(p)

    def _gen_move_cost_array(self, action):
        _action = abs(action)

        if _action != 0:
          _action -= 1

        # Don't punish bad action:
        if self.bad_action_effect == 0:
            cost = np.asarray(
                [ii if ii < _action else _action for ii in range(self.capacity+1)]
            ) * self.cost_of_moving

        # bad action is punished
        else:
            cost = np.asarray(
                [self.bad_action_effect if ii < _action else _action for ii in range(self.capacity + 1)]
            ) * self.cost_of_moving
        return cost


    @classmethod
    def expected_rental_reward(cls, expected_request):
        return np.asarray([cls._state_reward(s, expected_request) for s in range(cls.capacity + 1)])

    @classmethod
    def _state_reward(cls, s, mu):
        rewards = cls.rental_reward * np.arange(s + 1)
        p = Poisson.pmf_series(mu, cutoff=s)
        return rewards.dot(p)

    def transition_probabilty(self, s, req, ret, action=0):

        _ret_sz = self.max_movement_allowed + self.capacity

        p_req = Poisson.pmf_series(req, s)
        p_ret = Poisson.pmf_series(ret, _ret_sz)
        p = np.outer(p_req, p_ret)

        transp = np.asarray([p.trace(offset) for offset in range(-s, _ret_sz + 1)])

        assert abs(action) <= self.max_movement_allowed, "action can be large than %s." % self.max_movement_allowed

        # No GBikes are being moved
        if action == 0:
            transp[20] += sum(transp[21:])
            return transp[:21]

        # Move GBikes from station 1 to station 2
        if action > 0:
            transp[self.capacity-action] += sum(transp[self.capacity-action+1:])
            transp[self.capacity-action+1:] = 0

            return np.roll(transp, shift=action)[:self.capacity+1]

        # Move GBikes from station 2 to station 1
        action = -action
        transp[action] += sum(transp[:action])
        transp[:action] = 0

        transp[action+self.capacity] += sum(transp[action+self.capacity+1:])
        transp[action+self.capacity+1:] = 0

        return np.roll(transp, shift=-action)[:self.capacity+1]

    def policy_iteration(self):
        self.policy_evaluation()
        while self.policy_update():
            self.policy_evaluation()


solver = PolicyIS()

for ii in range(4):
    solver.policy_evaluation()
    solver.policy_update()

print(solver.policy)

import matplotlib.pylab as plt

plt.subplot(131)
CS = plt.contour(solver.policy, levels=range(-6, 7), colors='r')
plt.clabel(CS, fontsize=10, inline=True)
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.xticks(range(0, 21, 4))
plt.yticks(range(0, 21, 4))
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

plt.subplot(132)
plt.pcolor(solver.value, cmap='Blues')
plt.colorbar()
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.xticks(range(0, 21, 4))
plt.yticks(range(0, 21, 4))
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)


plt.tight_layout()
plt.show()
