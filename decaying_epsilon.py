import numpy as np
import random
from automated_gridworld import AutoGridWorld as AGW

# Epsilon greedy method

def epsilon_greedy(values, state, action_space, epsilon = 0.1):
  # get a random number between 0 and 1 in uniform distribution 
  prob = np.random.random()
  # return a random choice if the prob is less than epsilon
  if prob < epsilon:
      return np.random.choice(action_space)
  else:
      return max(values[state], key = values[state].get) 

# Generate grid
grid = AGW(rows=5, columns=5, num_walls=1, start=(0, 0), step_cost=-0.01)
print("Grid world generated")
grid.print_grid()

ActionSpace = grid.action_space
ConvergenceThreshold = 0.001
Alpha = 0.3
Gamma = 0.9

# Initialize State Values
V = {}
for s in grid.actions.keys():
  # assign 0 for each state
  V[s] = {}
  for a in ActionSpace:
    V[s][a] = 0

deltas = []

# Enter the number of episodes to play
max_episodes = 10000
deltas = []
for iter in range(max_episodes):
  # set a random starting position
  s = grid.random_reset()
  delta = 0  
  state_visited = []
  # print("episode", iter)
  while not grid.game_over():
    # get the action based on rl method
    a = epsilon_greedy(values = V, state = s, action_space=ActionSpace, epsilon= 1 - (iter/max_episodes) )
    r = grid.move(a)
    s_next = grid.current_state()
    # update value of the state only for first visit
    if True: #s not in state_visited:
      state_visited.append(s)
      v_old = V[s][a]
      # Modified bellman's equation
      # Alpha is the factor for learning like exponential moving avg
      # Gamma is forgetting factor in bellman's equation 
      V[s][a] = V[s][a] + Alpha * (r + Gamma * max(V.get(s_next, {0:0}).values()) - V[s][a])
      delta = max([delta, np.abs(V[s][a] - v_old)])
    s = s_next

  deltas.append(delta)

  if iter > max_episodes/10 and np.mean(deltas[-50]) < ConvergenceThreshold:
    print(f"Learning stopped after {iter + 1} episodes")
    break


# Get best policy
policy_best = {}
for i in range(grid.rows):
  for j in range(grid.cols):
    # assign 0 for each state
    if (i, j) in V.keys():
      policy_best[(i, j)] = max(V[(i, j)], key = V[(i, j)].get)
    else:
      policy_best[(i, j)] = "x"

print("Grid world generated")
grid.print_grid()
print('\n')
print("Best Policy identified")
grid.print_policy(policy_best)

plt.plot(deltas)
plt.title("Learning per episode")
plt.xlabel("Episode")
plt.ylabel("Delta")