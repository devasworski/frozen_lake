<img src="https://people.bath.ac.uk/mtc47/img/collaborators/QM_Logo.png" height=100>

## Artificial Intelligence in Games (Assignment 3): 
# Frozen Lake

Module Code: **ECS7002P** 

Module Leader: **Diego Perez-Liebana**

Semester: **1**

Submission Date: **10th Januaray 2022**

__Team Members:__
* [Alexander Sworski](mailto:a.sworski@se21.qmul.ac.uk)
* [Dimitrios Mylonas](mailto:d.mylonas@se21.qmul.ac.uk)
* [Amir Sepehr Aminian](mailto:a.aminian@se21.qmul.ac.uk)

#### Final Grade: 98/100

## Tasks

### Task 1: Enviroment
The frozen lake environment has two main variants: the small frozen lake (4x4) and the big frozen lake (8x8). In both cases, each tile in a square grid corresponds to a state. There is also an additional absorbing state, which will be introduced soon. There are four types of tiles: start (grey), frozen lake (light blue), hole (dark blue), and goal (white). The agent has four actions, which correspond to moving one tile up, left, down, or right. However, with probability 0.1, the environment ignores the desired direction and the agent slips (moves one tile in a random direction, which may be the desired direction). An action that would cause the agent to move outside the grid leaves the state unchanged.

<img width="268" alt="Screenshot 2022-03-24 at 12 38 43" src="https://user-images.githubusercontent.com/34026653/159917907-e2864a2d-56a0-469e-ac19-2296e188e20e.png"><img width="267" alt="Screenshot 2022-03-24 at 12 39 07" src="https://user-images.githubusercontent.com/34026653/159917978-40bb8c73-ddbe-4b15-a87a-2d2ea99e27ed.png">



The agent receives reward 1 upon taking an action at the goal. In every other case, the agent receives zero reward. Note that the agent does not receive a reward upon moving into the goal (nor a negative reward upon moving into a hole). Upon taking an action at the goal or in a hole, the agent moves into the absorbing state. Every action taken at the absorbing state leads to the absorbing state, which also does not provide rewards. Assume a discount factor of γ = 0.9.

For the purposes of model-free reinforcement learning (or interactive testing), the agent is able to interact with the frozen lake for a number of time steps that is equal to the number of tiles.
#### Goal:
        + Implement the Enviroment in Python  
#### Code:
        - frozen_lake.py
### Task 2: Tabular model-based reinforcement learning
#### Goal:
        + Implement policy evaluation
        + Implement policy improvement
        + Implement policy iteration
        + Implement value iteration
#### Code:
        - model_based_rl.py
### Task 3: Tabular model-free reinforcement learning
#### Goal:
        + Implement Sarsa control 
        + Implement Q-learning control
#### Code:
        - tabular_model_free_rl.py
### Task 4: Non-tabular model-free reinforcement learning
#### Goal:
        + Implement Sarsa control using linear function approximation
        + Implement Q-learning control using linear function approximation
#### Code:
        - non_tabular_model_free_rl.py
### Task 5: Main function
#### Goal:
        + Implement a main function to execute all tasks
#### Code:
        - flake.py


# man
The man page for the flake implementation.

##### **NAME**
        flake - the rl game

##### **SYNOPSIS**
        python ./flake.py [-T <task_number>] [-s] [-l] [-v]

##### **DESCRIPTION**
        flake is a small CLI game for reinforcement learning based agents.
        Multiple rl methodes can be used for the agents

##### **OPTIONS**
        -T <task_number>
           Run flake as execute the given Task, represented by the number 2 to 6.
           By deflaut task 5 is executed.
           2: Model based RL
           3: tabular model free RL
           4: non tabular model free RL
           5: all
           6: Itteration count to find optimal policy for tabular model free RL
        - s
            Run the program with a small lake (4x4)
        - l
            Run the program with a big lake (8x8)
        - v
            output the results as a .png file.
            the ghostscript library is required (instructions below).


#### **Output**
**Tile represenation**:

| Tile         | Icon |
|--------------|:-----:|
| start | & |
| frozen |  . |
| hole |  # |
| goal | $ |

<br>
<br>

#### **Ghostscript Installation:**

+ **For Mac:**\
`brew install ghostscript`

+ **For Windows:**\
Go on the official [Webpage](https://ghostscript.com/releases/) to download the installer
