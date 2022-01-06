## Task 5 ##

# Task imports
from frozen_lake import FrozenLake
from model_based_rl import policy_iteration, value_iteration
from tabular_model_free_rl import sarsa, q_learning
from non_tabular_model_free_rl import linear_q_learning, linear_sarsa, LinearWrapper
from gui_map import GameWindow
from tkinter import Tk
#general imports
import numpy as np
import sys, getopt

''' execute function
    executes all RL methods or one specific and shows the policies that would result
    
    @param task
        @default = 5
        the task that should be executed
    @param lake_size
        @default = s
        the lake size small (s) or large (l)
'''
def execute(task=5, lake_size='s', visual=False):

    task = int(task)
    if task<2 or task>6:
        print('WRONG INPUT')
        exit(1)


    seed = 0
    gamma = 0.9
    theta = 0.001
    max_iterations = 10000
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    # Small lake
    small_lake =    [['&', '.', '.', '.'],
                    ['.', '#', '.', '#'],
                    ['.', '.', '.', '#'],
                    ['#', '.', '.', '$']]

    big_lake =      [['&', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '#', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '#', '.', '.'],
                    ['.', '.', '.', '#', '.', '.', '.', '.'],
                    ['.', '#', '#', '.', '.', '.', '#', '.'],
                    ['.', '#', '.', '.', '#', '.', '#', '.'],
                    ['.', '.', '.', '#', '.', '.', '.', '$']]

    print('')
    print('')

    if lake_size == 's':
        lake = small_lake
        print('Using small lake')
    elif lake_size == 'l':
        lake = big_lake
        print('Using big lake')
    else:
        print('WRONG INPUT')
        exit(1)

    if visual:
        root = Tk()
        g = GameWindow(root,lake)

    size = len(lake) * len(lake[0])
    env = FrozenLake(lake, slip=0.1, max_steps=size, seed=seed)
    
    print('')
    print('')

    if task == 5 or task == 2:  
        print('# Model-based algorithms')

        print('')

        print('## Policy iteration')
        policy, value = policy_iteration(env, gamma, theta, max_iterations)
        env.render(policy, value)
        if visual:
            g.create_arrow_value_map(lake,value,policy)

        print('')

        print('## Value iteration')
        optimal_policy, value = value_iteration(env, gamma, theta, max_iterations)
        env.render(optimal_policy, value)
        if visual:
            g.create_arrow_value_map(lake,value,optimal_policy)

        print('')
        print('')

    if task == 5 or task == 3:  
        print('# Model-free algorithms')

        print('')

        print('## Sarsa')
        
        policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
        env.render(policy, value)
        if visual:
            g.create_arrow_value_map(lake,value,policy)

        print('')

        print('## Q-learning')
        policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
        env.render(policy, value)
        if visual:
            g.create_arrow_value_map(lake,value,policy)

        print('')
        print('')

    if task == 5 or task == 4:  
        linear_env = LinearWrapper(env)

        print('## Linear Sarsa')
        parameters = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
        policy, value = linear_env.decode_policy(parameters)
        linear_env.render(policy, value)
        if visual:
            g.create_arrow_value_map(lake,value,policy)

        print('')

        print('## Linear Q-learning')
        parameters = linear_q_learning(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
        policy, value = linear_env.decode_policy(parameters)
        linear_env.render(policy, value)
        if visual:
            g.create_arrow_value_map(lake,value,policy)

        print('')
        print('')

    if task == 5 or task==6:  
        print('# Additional outputs for the report:')

        print('')
        #TODO (unsure if this is really what has been asked):
        print('## iteration require to find an optimal policy')


        policy, value = policy_iteration(env, gamma, theta, max_iterations)
        optimal_policy, value = value_iteration(env, gamma, theta, max_iterations)

        for episodes in np.arange(500,6000,100):
            print(f'Sarsa episodes = {episodes}')
            policy, value = sarsa(env, episodes, eta, gamma, epsilon, seed=seed)
            if np.array_equal(policy, optimal_policy):
                break
        env.render(policy, value)

        print('')
        for episodes in np.arange(500,6000,100):
            print(f'Q-learning episodes = {episodes}')
            policy, value = q_learning(env, episodes, eta, gamma, epsilon, seed=seed)
            if np.array_equal(policy, optimal_policy):
                break
        env.render(policy, value)
    if visual:
        root.mainloop()
''' main_args function
    takes the commandline parameters and passes them over to the execute function
    
    @param argv
        command line arguments
'''
def main_args(argv):
    task = 5
    lake_size = 's'
    visual = False
    try:
        opts, args = getopt.getopt(argv,"T:slv",[])
    except getopt.GetoptError:
        execute()
    for opt, arg in opts:
        if opt == '-T':
            task = int(arg)
        elif opt in ("-s"):
            lake_size = 's'
        elif opt in ("-l"):
            lake_size = 'l'
        elif opt in ("-v"):
            visual = True

    execute(task,lake_size,visual)

if __name__ == "__main__":
   main_args(sys.argv[1:])
