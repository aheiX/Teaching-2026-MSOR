def get_data_textbook():
    """
    Creates the model-based data for the service dog example used in "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        states: The set of all possible configurations or observations of the environment that we can be in.
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    """
    
    # State space [𝒮]: The set of all possible configurations or observations of the environment that we can be in.
    states = ["Room 1", "Room 2", "Room 3", "Outside", "Found item"]


    # Reward [R(s,a)]: Reward of taking action 𝑎 in state 𝑠 
    reward = {("Room 1", "Go to room 2"): -1,
                ("Room 2", "Go to room 1"): -2,
                ("Room 2", "Go to room 3"): -1,
                ("Room 2", "Go outside"): 0,
                ("Room 3", "Go to room 2"): -2,
                ("Room 3", "Search"): 10,
                ("Outside", "Go outside"): -1,    # =-1 in textbook figure, but =-2 in textbook GitHub code (and used for textbook values)
                ("Outside", "Go inside"): 0              
                }

    # Transition probability [P(s'|s,a)]: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    # depends on the current state 𝑠 and the action 𝑎 chosen by the agent. There may be multiple successor states.
    transition_prob = {
        ("Room 1", "Go to room 2"): {"Room 2": 1.0},
        ("Room 2", "Go to room 1"): {"Room 1": 1.0},
        ("Room 2", "Go to room 3"): {"Room 3": 1.0},
        ("Room 2", "Go outside"): {"Outside": 1.0},
        ("Room 3", "Go to room 2"): {"Room 2": 1.0},
        ("Room 3", "Search"): {"Found item": 1.0},
        ("Outside", "Go outside"): {"Outside": 1.0},   
        ("Outside", "Go inside"): {"Room 2": 1.0} 
    }

    return states, reward, transition_prob


def get_policy_random():
    """
    Creates a random policy for the service dog example used in "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        policy: The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
    """

    # Policy [𝜋(a|s)]: The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
    policy_random = {
        "Room 1": {"Go to room 2": 1.0},
        "Room 2": {"Go to room 1": 1/3, "Go to room 3": 1/3, "Go outside": 1/3},
        "Room 3": {"Go to room 2": 0.5, "Search": 0.5},
        "Outside": {"Go inside": 0.5, "Go outside": 0.5},
        "Found item": {}
    }

    return policy_random


def policy_evaluation(states, policy, reward, transition_prob, discount, delta_threshold=0.00001):
    """
    Given a policy function, reward function, transition probability function, discount factor and a delta threshold,
    using dynamic programming to estimate the state-value function for this policy.

    Args:
        states: State space 𝒮
        policy: Policy we want to evaluate. The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′. There may be multiple successor states.
        discount: discount factor, must be 0 <= discount <= 1.
        delta_threshold: the threshold determining the accuracy of the estimation.
    """

    # Initialize counter
    count = 0
    
    # Initialize state value function to be all zeros for all states.
    V = {s: 0 for s in states}

    while True:
        delta = 0
        for state in states:
            old_v = V[state]
            new_v = 0

            # For every legal action
            for action, action_prob in policy[state].items():   

                # Immediate reward
                g = reward[(state, action)]

                # Future reward                
                for successor_state, successor_state_prob in transition_prob[(state, action)].items():
                    # Note one state-action might have multiple successor states with different transition probability
                    # Weight by the transition probability
                    g += discount * successor_state_prob * V[successor_state]

                # Weight by the probability of selecting this action when following the policy
                new_v += action_prob * g
            V[state] = new_v
            delta = max(delta, abs(old_v - new_v))

        count += 1
        if delta < delta_threshold:
            for state in V:
                V[state] = round(V[state], 2)
            break
    
    print(f"State value function after {count} iterations: {V}")


def value_iteration(states, reward, transition_prob, discount, delta_threshold=0.00001):
    """
    Given a policy function, reward function, transition probability function, discount factor and a delta threshold,
    find a optimal policy 𝜋* along with optimal state value function V*.

    Args:
        states: State space 𝒮
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′. There may be multiple successor states.
        discount: discount factor, must be 0 <= discount <= 1.
        delta_threshold: the threshold determining the accuracy of the estimation.
    """

    # Initialize counter
    count = 0

    # Initialize & retrieve legal actions from reward function
    legal_actions = {state: [] for state in states}
    for (state, action) in reward.keys():
        legal_actions[state].append(action)

    # Initialize state value function to be all zeros for all states.
    V = {s: 0 for s in states}

    while True:
        delta = 0
        for state in states:
            old_v = V[state]
            # Store the expected returns for each action.
            estimated_returns = [0]

            # For every legal action
            for action in legal_actions[state]:   
                               
                # Immediate reward
                g = reward[(state, action)]

                # Future reward                
                for successor_state, successor_state_prob in transition_prob[(state, action)].items():
                    # Note one state-action might have multiple successor states with different transition probability
                    # Weight by the transition probability
                    g += discount * successor_state_prob * V[successor_state]

                estimated_returns.append(g)

            # Use the maximum expected returns across all actions as state value.
            V[state] = max(estimated_returns)
            delta = max(delta, abs(old_v - V[state]))

        count += 1
        if delta < delta_threshold:            
            break

    # Step 2: Compute an optimal policy from optimal state value function.
    optimal_policy = {s: dict() for s in states}
    for state in states:
        # Store the expected returns for each action.
        estimated_returns = {}

        # For every legal action
        for action in legal_actions[state]:   
                            
            # Immediate reward
            g = reward[(state, action)]

            # Future reward                
            for successor_state, successor_state_prob in transition_prob[(state, action)].items():
                # Note one state-action might have multiple successor states with different transition probability
                # Weight by the transition probability
                g += discount * successor_state_prob * V[successor_state]

            estimated_returns[action] = g
        
        if len(estimated_returns) > 0:
            # Get the best action a based on the q(s, a) values, notice the action is the key in the dict estimated_returns.
            best_action = max(estimated_returns, key=estimated_returns.get)

            # Set the probability to 1.0 for the best action.
            optimal_policy[state][best_action] = 1.0

    # Round state value function
    for state in V:
        V[state] = round(V[state], 2)

    print(f"State value function after {count} iterations: {V}")
    print(f"Optimal policy after {count} iterations: {optimal_policy}")


if __name__ == "__main__":
    # Load data
    states, reward, transition_prob = get_data_textbook()
    policy = get_policy_random()

    # Policy evaluation
    print("Policy evaluation (random policy)")
    policy_evaluation(states, policy, reward, transition_prob, discount=0.9)
    
    # Value iteration
    # print("Value iteration (optimal policy)")
    # value_iteration(states, reward, transition_prob, discount=0.9)
