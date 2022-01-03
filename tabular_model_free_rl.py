## Task 3 ##

import numpy as np

''' <name> Function TODO
    TODO: This was not part of the task description interface. Therefore we should remove it
    <explenation>
    
    @param <param name>
    <param description>
    
    @return <return param>
        <return param description>
'''
def randomBestAction(random_state, mean_rewards):
    # get the best actions from mean_rewards
    best_actions = np.array(np.argwhere(mean_rewards == np.amax(mean_rewards))).flatten()
    return random_state.choice(best_actions, 1)[0]  # break ties randomly and return one of the best actions

''' sarsa function TODO
    TODO <explenation>
    
    @param env
        current enviroment
    @param max_episodes
        maximum number if episodes
    @param eta
        initial learning rate
    @param gamma
        discount factor
    @param seed
        @default = None
        seed for random numbers
    
    @return policy
        the improved policy
    @return value
        the evaluation value of the policy
'''
def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes) 

    q = np.zeros((env.n_states, env.n_actions)) 

    #TODO:
    timestep = 0
    #END TODO

    for i in range(max_episodes):
        s = env.reset() 

        #TODO:
        
        #Select action a for state s according to an e-greedy policy based on Q
        if(timestep < env.n_actions):  # for the first 4 timesteps, choose each action once
            a = timestep  # select each action 0, 1, 2, 3 once
        else:
            # after having our first estimations, find the best action and break ties randomly
            best_action = randomBestAction(random_state, q[s])

            # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
            # (exploitation) or a random action (exploration)
            if(random_state.random(1) < epsilon[i]):
                a = random_state.choice(range(env.n_actions))  # use random action
            else:
                a = best_action  # use best action
        timestep += 1

        done = False
        while(not done):  # while not in absorbing state
            s_prime, r, done = env.step(a)

            #Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if(timestep < env.n_actions):  # for the first 4 timesteps, choose each action once
                a_prime = timestep  # select each action 0, 1, 2, 3 once
            else:
                # after having our first estimations, find the best action and break ties randomly
                best_action = randomBestAction(random_state, q[s_prime])

                # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
                # (exploitation) or a random action (exploration)
                if(random_state.random(1) < epsilon[i]):
                    a_prime = random_state.choice(range(env.n_actions))  # use random action
                else:
                    a_prime = best_action  # use best action
            timestep += 1

            # update estimated value of the current state and action
            q[s,a] += eta[i] * (r + gamma * q[s_prime, a_prime] - q[s,a])
            s = s_prime
            a = a_prime

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

''' q_learning Function TODO
    TODO <explenation>
    
    @param env
        current enviroment
    @param max_episodes
        maximum number if episodes
    @param eta
        initial learning rate
    @param gamma
        discount factor
    @param seed
        @default = None
        seed for random numbers
    
    @return policy
        the improved policy
    @return value
        the evaluation value of the policy
'''
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions)) 
    
    #TODO:
    timestep = 0
    #END TODO

    for i in range(max_episodes):
        s = env.reset()  

        #TODO:

        done = False
        while(not done):  # while not in absorbing state

            #Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if(timestep < env.n_actions):  # for the first 4 timesteps, choose each action once
                a = timestep  # select each action 0, 1, 2, 3 once
            else:
                # after having our first estimations, find the best action and break ties randomly
                best_action = randomBestAction(random_state, q[s])

                # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
                # (exploitation) or a random action (exploration)
                if(random_state.random(1) < epsilon[i]):
                    a = random_state.choice(range(env.n_actions))  # use random action
                else:
                    a = best_action  # use best action
            timestep += 1

            s_prime, r, done = env.step(a)  # Get next state and reward for the chosen action

            q_max = max(q[s_prime])  # find the best action for next step
            # update estimated value of the current state and action
            q[s,a] += eta[i] * (r + gamma * q_max - q[s,a])
            s = s_prime


    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value