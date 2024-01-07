import numpy as np
import random
from automated_gridworld import AutoGridWorld as AGW
from sklearn.kernel_approximation import Nystroem, RBFSampler
import matplotlib.pyplot as plt

# Generate grid
grid = AGW(rows=5, columns=5, num_walls=1, start=(0, 0), step_cost=-0.01)
print("Grid world generated")
grid.print_grid()

ActionSpace = grid.action_space
Gamma = 0.9
Alpha = 0.1
Action2Int = {a:int(i) for i, a in enumerate(ActionSpace)}
Int2OneHot = np.eye(len(ActionSpace), dtype=int)

class LinearModel:
  def __init__(self, grid, sampler = "RBF"):
    self.actions = grid.actions
    samples = self.gather_initial_samples(grid, 500)

    num_dimensions = 100

    if sampler == "RBF":
      self.featurizer = RBFSampler(n_components= num_dimensions) 
    elif sampler == "Nystroem":
      self.featurizer = Nystroem(n_components= num_dimensions)
    
    self.featurizer.fit(samples)

    # initilize weights 
    self.weights = np.zeros(self.featurizer.n_components)

  def action_to_onehot(self, action):
    action_int = Action2Int[action]
    action_one_hot = Int2OneHot[action_int]
    return action_one_hot

  def merge_state_action(self, state, action):
    action_onehot = self.action_to_onehot(action)
    merged = np.concatenate((state, action_onehot))
    return merged

  def gather_initial_samples(self, grid, n_episodes = 100):
    samples = []
    for _ in range(n_episodes):
      s = grid.random_reset()
      steps = 0
      while not (grid.game_over() or steps > 20):
        allowed_actions = self.actions[s]
        a = random.choice(allowed_actions)
        # print(s, allowed_actions, a)
        state_action = self.merge_state_action(s, a)
        samples.append(state_action)
        r = grid.move(a)

        s = grid.current_state()
        steps += 1
    return samples

  def feature_transform(self, state, action):
    state_action = self.merge_state_action(state, action)
    features = self.featurizer.transform([state_action])[0]
    
    return features

  def predict(self, state, action):
    input = self.feature_transform(state, action)
    return input @ self.weights

  def predict_all_actions(self, state):
    predictions = []
    for action in ActionSpace:
      predictions.append(self.predict(state, action))

    return predictions

  def soft_epsilon_greedy(self, state, eps = 0.1):
    p = np.random.random()
    if p > eps:
      predictions = self.predict_all_actions(state)
      best_action = np.argmax(predictions)
      return ActionSpace[best_action]
    else:
      allowed_actions = self.actions[s]
      return random.choice(allowed_actions)

if __name__ == "__main__":
  lmodel = LinearModel(grid)
  reward_per_episode = []
  state_visit_count = {}

  n_episode = 1000
  for iter in range(n_episode):
    reward = 0
    s = grid.random_reset()
    state_visit_count[s] = state_visit_count.get(s, 0) + 1
    while not grid.game_over():
      a = lmodel.soft_epsilon_greedy(s)
      r = grid.move(a)
      s2 = grid.current_state()
      state_visit_count[s2] = state_visit_count.get(s2, 0) + 1

      # get target
      if grid.game_over():
        target = r
      else:
        values = lmodel.predict_all_actions(s2)
        target = r + Gamma * np.max(values)
      
      # model update 
      grad = lmodel.feature_transform(s, a)
      err = target - lmodel.predict(s, a)   # this is the error in prediction
      lmodel.weights += Alpha * err * grad

      # accumulate reward
      reward += r

      # state update 
      s = s2

    reward_per_episode.append(reward)

  plt.plot(reward_per_episode)
  plt.title("Reward per episode")
  plt.show()

  # Get best policy
  policy_best = {}
  for i in range(grid.rows):
    for j in range(grid.cols):
      if (i, j) in grid.actions.keys():
        preds = lmodel.predict_all_actions((i, j))
        best_action = ActionSpace[np.argmax(preds)]
        policy_best[(i, j)] = best_action[0]
      else:
        policy_best[(i, j)] = "X"

  print("Grid world generated")
  grid.print_grid()
  print('\n')
  print("Best Policy learned")
  grid.print_policy(policy_best)  

