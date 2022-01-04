## Task 3 ##

import numpy as np

''' get_actions function
    get the next action for the sarsa function

    @param env
        current enviroment
    @param q
        q-values
    @param epsilon
        linespace
    @param s
        current state
    @param random_state
        randomState Object

    @return state_a
        the next action
'''
def get_action(random_state,epsilon,i,env,q,s):
    if(random_state.random(1) < epsilon[i]):
        state_a = random_state.choice(range(env.n_actions))
    else:
        state_a = random_state.choice(np.array(np.argwhere(q[s] == np.amax(q[s]))).flatten(), 1)[0] 
    return state_a

''' sarsa function
    using the sarsa algorithem to get the q-values and policy
    
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

    for i in range(max_episodes):
        s = env.reset() 
        done = False
        if(i < env.n_actions):
            a = i
        else:
            a = get_action (random_state,epsilon,i,env,q,s)
        while(not done):
            state_s, reward_pre, done = env.step(a)
            
            state_a = get_action(random_state,epsilon,i,env,q,s)

            q[s,a] += eta[i] * ((reward_pre + gamma * q[state_s, state_a]) - q[s,a])
            a = state_a
            s = state_s

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