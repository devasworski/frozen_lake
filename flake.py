## Task 5 ##
# Task imports
from frozen_lake import FrozenLake
from model_based_rl import policy_iteration, value_iteration
from tabular_model_free_rl import sarsa, q_learning
from non_tabular_model_free_rl import linear_q_learning, linear_sarsa, LinearWrapper

#general imports
import numpy as np

''' main function
    executes all RL methodes or one sepcific and shows the policies that would results
    
    @param task
    @default = 5
    the task that should be executed
    @param lake_size
    @default = s
    the lake size small (s) or large (l)
'''
def main(task=5, lake_size='s'):
    seed = 0

    # Small lake
    small_lake =    [['&', '.', '.', '.'],
                    ['.', '#', '.', '#'],
                    ['.', '.', '.', '#'],
                    ['#', '.', '.', '$']]

    big_lake =      [['&', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '#', '.', '.'],
                    ['.', '.', '#', '.', '#', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '.', '.', '.'],
                    ['#', '.', '.', '#', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '#', '.', '#', '.'],
                    ['.', '#', '.', '#', '#', '.', '#', '.'],
                    ['.', '.', '.', '.', '.', '.', '.', '$']]

    if lake_size == 's':
        lake = small_lake
    else:
        lake = big_lake
    
    size = len(lake) * len(lake[0])
    env = FrozenLake(lake, slip=0.1, max_steps=size, seed=seed)

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 10000

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    optimal_policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(optimal_policy, value)

    print('')
    print('')

    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    print('')

    print('## Sarsa')
    
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')
    print('')

    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')
    parameters = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')
    parameters = linear_q_learning(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')
    print('')
    print('# Additional outputs for the report:')

    #TODO:
    print('## iteration require to find an optimal policy')

    for episodes in np.arange(500,5000,100):
        print(f'sarsa episodes = {episodes}')
        policy, value = sarsa(env, episodes, eta, gamma, epsilon, seed=seed)
        policy[15] = 0 
        if np.array_equal(policy, optimal_policy):
            break
    env.render(policy, value)

    for episodes in np.arange(500,5000,100):
        print(f'q_learning episodes = {episodes}')
        policy, value = q_learning(env, episodes, eta, gamma, epsilon, seed=seed)
        policy[15] = 0 
        if np.array_equal(policy, optimal_policy):
            break
    env.render(policy, value)

    print('## iteration require to find an optimal policy for big lake')

    for episodes in np.arange(500,5000,100):
        print(f'sarsa episodes = {episodes}')
        policy, value = sarsa(env, episodes, eta, gamma, epsilon, seed=seed)
        policy[15] = 0 
        if np.array_equal(policy, optimal_policy):
            break
    env.render(policy, value)

    for episodes in np.arange(500,5000,100):
        print(f'q_learning episodes = {episodes}')
        policy, value = q_learning(env, episodes, eta, gamma, epsilon, seed=seed)
        policy[15] = 0 
        if np.array_equal(policy, optimal_policy):
            break
    env.render(policy, value)


main()
