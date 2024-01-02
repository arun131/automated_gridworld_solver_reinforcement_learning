import random

class AutoGridWorld:
  def __init__(self, rows, columns, num_walls, start, step_cost = -0.1):
      self.i = start[0]
      self.j = start[1]
      self.start = start
      self.rows = rows
      self.cols = columns
      self.step_cost = step_cost
      self.grid, self.actions, self.rewards, self.terminal_states = self.generate_grid(rows, columns, num_walls, start)
      self.action_space = ["up", "down", "left", "right"]
      self.current_start = self.start

  def set_state(self, s):
      self.i = s[0]
      self.j = s[1]

  def current_state(self):
      return (self.i, self.j)

  def is_terminal(self, s):
      return s in self.terminal_states

  def reset(self):
      # put agent back in start position
      self.i = self.start[0]
      self.j = self.start[1]
      return (self.i, self.j)

  def random_reset(self):
      # put agent back in a random start position
      ran_start = random.choice(list(self.actions.keys()))
      self.i, self.j = ran_start[0], ran_start[1] 
      self.current_start = (self.i, self.j)
      return (self.i, self.j)

  def get_next_state(self, s, a):
    # this answers: where would I end up if I perform action 'a' in state 's'?
    i, j = s[0], s[1]

    # if this action moves you somewhere else, then it will be in this dictionary
    if a in self.actions[(i, j)]:
      if a == 'up':
        i -= 1
      elif a == 'down':
        i += 1
      elif a == 'right':
        j += 1
      elif a == 'left':
        j -= 1
    return i, j

  def move(self, action):
    # check if legal move first
    if action in self.actions[(self.i, self.j)]:
      if action == 'up':
        self.i -= 1
      elif action == 'down':
        self.i += 1
      elif action == 'right':
        self.j += 1
      elif action == 'left':
        self.j -= 1
    # return a reward (if any)
    return self.rewards.get((self.i, self.j), 0)

  def undo_move(self, action):
    # these are the opposite of what U/D/L/R should normally do
    if action == 'up':
      self.i += 1
    elif action == 'down':
      self.i -= 1
    elif action == 'right':
      self.j -= 1
    elif action == 'left':
      self.j += 1
    # raise an exception if we arrive somewhere we shouldn't be
    # should never happen
    assert(self.current_state() in self.grid.keys())

  def game_over(self):
    return (self.i, self.j) in self.terminal_states

  def generate_grid(self, rows, columns, num_walls, start):
      # Initialize the grid with all zeros
      grid = [[0] * columns for _ in range(rows)]
      blocks = [tuple(start)]
      # Set random walls
      for _ in range(num_walls):
          row = random.randint(0, rows - 1)
          col = random.randint(0, columns - 1)
          grid[row][col] = -1  # -1 represents a wall
          blocks.append((row, col))
      rewards = {}
      # Set winning and losing blocks
      winning_block = ()
      while winning_block in blocks or winning_block == ():
        winning_block = (random.randint(0, rows - 1), random.randint(0, columns - 1))
      blocks.append(winning_block)


      losing_block = ()
      while losing_block in blocks or losing_block == ():
        losing_block = (random.randint(0, rows - 1), random.randint(0, columns - 1))
      blocks.append(losing_block)


      grid[winning_block[0]][winning_block[1]] = 1  # 1 represents the winning block
      rewards[winning_block] = 1
      grid[losing_block[0]][losing_block[1]] = -2  # -2 represents the losing block
      rewards[losing_block] = -2

      # Define allowed actions for each position in the grid
      allowed_actions = {}
      for row in range(rows):
          for col in range(columns):
              if grid[row][col] not in [-1, 1, -2]:  # Exclude walls, winning, and losing blocks
                  # Determine allowed actions based on the position
                  actions = []
                  if row > 0 and grid[row - 1][col] not in [-1]:
                      actions.append("up")
                  if row < rows - 1 and grid[row + 1][col] not in [-1]:
                      actions.append("down")
                  if col > 0 and grid[row][col - 1] not in [-1]:
                      actions.append("left")
                  if col < columns - 1 and grid[row][col + 1] not in [-1]:
                      actions.append("right")

                  allowed_actions[(row, col)] = actions
                  if not (row, col) == tuple(start):
                    rewards[(row,col)] = self.step_cost

      return grid, allowed_actions, rewards, [winning_block, losing_block]

  def print_grid(self):
    for i, row in enumerate(self.grid):
      r = ""
      for j, val in enumerate(row):
        v = val if (i, j) != self.current_start else "x"
        r += f" {v}|" if val >= 0 else f"{v}|"
      print(r)
      print("-----------------")

  def print_policy(self, policy):
    for i, row in enumerate(self.grid):
      r = ""
      for j, val in enumerate(row):
        v = policy.get((i,j), '0')[0]
        r += f" {v}|"
      print(r)
      print("-----------------")


  def random_policy(self):
    policy = {}
    for key, item in self.actions.items():
      policy[key] = random.choice(item)
    return policy 
