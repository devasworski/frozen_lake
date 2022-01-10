## Task 3 ##

import numpy as np

''' get_actions function
    get the next action for the sarsa and q-learning function

    @param env
        current environment
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
    using the sarsa algorithm to get the q-values and policy
    
    @param env
        current environment
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
        expected total reward starting from each game state
'''
def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes) 

    q = np.zeros((env.n_states, env.n_actions)) 

    for i in range(max_episodes):
        s = env.reset() 
        done = False
        j = 0
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

''' q_learning Function
    using the q-learning algorithm which is off-policy to find optimal policy
    
    @param env
        current environment
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
        expected total reward starting from each game state
'''
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions)) 

    for i in range(max_episodes):
        s = env.reset()  
        j = 0
        done = False
        while(not done):

            if(j < env.n_actions):
                a = j
            else:
                a = get_action(random_state,epsilon,i,env,q,s)
            j += 1

            state_s, reward_pre, done = env.step(a)

            q[s,a] += eta[i] * (reward_pre + gamma * max(q[state_s]) - q[s,a])
            s = state_s

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value