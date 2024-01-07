# automated_gridworld_solver_reinforcement_learning
Welcome to the Automated Grid World Reinforcement Learning Repository!

Whether you are a beginner seeking to understand the basics of reinforcement learning or an experienced practitioner experimenting with different environments, this script lays the groundwork for dynamic and automated grid world generation.

This Git repository houses essential Python scripts designed to facilitate the exploration and understanding of reinforcement learning algorithms within a dynamic grid world environment.

**automated_gridworld.py**:
This script encapsulates the AutomatedGridWorld class, providing a versatile implementation for generating random grid world environments. With the ability to specify the number of rows, columns, and the desired number of walls, this class serves as a foundation for creating diverse and customizable grid world scenarios.

Example Grid world generated:

 x|-1| 0| 0| 0|
-----------------
 0| 0| 0| 0| 0|
-----------------
 0| 0| 0| 0| 0|
-----------------
 0| 0| 0|-1|-1|
-----------------
 0|-2| 1| 0| 0|
-----------------

x is the starting position
-1 is a wall and the agent cannot traverse through the walls
-2 is a losing position.
1 is a winning position.

A random grid world is generated every time the code is executed.

**q_learning_decaying_epsilon_greedy.py**:
In this script, I present an implementation of the decaying epsilon-greedy algorithm for reinforcement learning. The decaying epsilon-greedy strategy strikes a balance between exploration and exploitation, crucial for training reinforcement learning agents. Executing several episodes of reinforcement learning in the automatically generated grid world created by automated_gridworld.py, this script showcases the algorithm's effectiveness. As a result, it prints the optimum policy, shedding light on the intelligent decision-making process developed by the agent over the course of training.

**machine_learning_rbf_kernal.py**:
In this script, I present an implementation of the machine learning approximation of the value function. I use RBFSampler for feature generation in this implementation. While action space exploration is done by soft epsilon-greedy strategy. Executing several episodes of reinforcement learning in the automatically generated grid world created by automated_gridworld.py, this script showcases the algorithm's effectiveness. As a result, it prints the optimum policy, shedding light on the intelligent decision-making process developed by the agent over the course of training.

Best policy identified:

 d| x| d| d| d|
-----------------
 d| d| d| d| d|
-----------------
 d| d| d| l| l|
-----------------
 r| r| d| x| x|
-----------------
 u| x| x| l| l|
-----------------
The script outputs the best move at any given block.
The possible action space is up, down, left, and right represented as u, d, l, and r respectively.

Feel free to explore, experiment, and build upon these scripts to deepen your understanding of reinforcement learning in grid world environments. Whether you're a researcher, student, or hobbyist, this repository aims to be a valuable resource for hands-on learning and experimentation in the field of reinforcement learning. Happy coding!
