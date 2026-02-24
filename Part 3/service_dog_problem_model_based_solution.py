
def get_data_task_2():
    """
    Creates the model-based data for the service dog problem based on "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        states: The set of all possible configurations or observations of the environment that we can be in.
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    """
    
    # State space [𝒮]: The set of all possible configurations or observations of the environment that we can be in.
    states = ["Room 1", "Room 2", "Room 3", "Outside", "Found item"]

    # Reward [R(s,a)]: Reward of taking action 𝑎 in state 𝑠 
    reward = {("Room 1", "Go to room 2"): 1,
              ("Room 2", "Go to room 1"): -2,
              ("Room 2", "Go to room 3"): 3,
              ("Room 2", "Go outside"): -2,
              ("Room 3", "Go to room 2"): -4,
              ("Room 3", "Search"): 12,
              ("Outside", "Go outside"): 1,  
              ("Outside", "Go inside"): 3              
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


def get_data_task_3():
    """
    Creates the model-based data for the service dog example based on "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        states: The set of all possible configurations or observations of the environment that we can be in.
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    """
    
    # State space [𝒮]: The set of all possible configurations or observations of the environment that we can be in.
    states = ["Room 1", "Room 2", "Room 3", "Room 4", "Outside", "Found item"]

    # Reward [R(s,a)]: Reward of taking action 𝑎 in state 𝑠 
    reward = {("Room 1", "Go to room 1"): -1,
              ("Room 1", "Go to room 2"): -1,
              ("Room 1", "Go to room 4"): 1,
              #
              ("Room 2", "Go to room 3"): -1,
              ("Room 2", "Go outside"): 1,
              # 
              ("Room 3", "Go outside"): 1,
              ("Room 3", "Search"): 10,
              # 
              ("Room 4", "Search"): 10,
              # 
              ("Outside", "Go inside"): 0   
              }

    # Transition probability [P(s'|s,a)]: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    # depends on the current state 𝑠 and the action 𝑎 chosen by the agent. There may be multiple successor states.
    transition_prob = {
        ("Room 1", "Go to room 1"): {"Room 1": 1.0},
        ("Room 1", "Go to room 2"): {"Room 2": 1.0},
        ("Room 1", "Go to room 4"): {"Room 4": 1.0},
        #
        ("Room 2", "Go to room 3"): {"Room 3": 1.0},
        ("Room 2", "Go outside"): {"Outside": 1.0},
        # 
        ("Room 3", "Go outside"): {"Outside": 1.0},
        ("Room 3", "Search"): {"Found item": 1.0},
        # 
        ("Room 4", "Search"): {"Found item": 1.0},
        # 
        ("Outside", "Go inside"): {"Room 2": 1.0}  
        }
    
    return states, reward, transition_prob


def get_data_task_4_and_5():
    """
    Creates the model-based data for the service dog example based on "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        states: The set of all possible configurations or observations of the environment that we can be in.
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    """
    
    # State space [𝒮]: The set of all possible configurations or observations of the environment that we can be in.
    states = ["Room 1", "Room 2", "Room 3", "Room 4", "Outside", "Found item"]

    # Reward [R(s,a)]: Reward of taking action 𝑎 in state 𝑠 
    reward = {("Room 1", "Go to room 1"): -1,
              ("Room 1", "Go to room 2"): -1,
              ("Room 1", "Go to room 4"): 1,
              #
              ("Room 2", "Go to room 3"): -1,
              ("Room 2", "Go outside"): 1,
              # 
              ("Room 3", "Go outside"): 1,
              ("Room 3", "Search"): 10,
              # 
              ("Room 4", "Search"): 10,
              # 
              ("Outside", "Go inside"): 0   
              }

    # Transition probability [P(s'|s,a)]: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    # depends on the current state 𝑠 and the action 𝑎 chosen by the agent. There may be multiple successor states.
    transition_prob = {
        ("Room 1", "Go to room 1"): {"Room 1": 0.8, "Outside": 0.2},
        ("Room 1", "Go to room 2"): {"Room 2": 0.8, "Outside": 0.2},
        ("Room 1", "Go to room 4"): {"Room 4": 0.8, "Outside": 0.2},
        #
        ("Room 2", "Go to room 3"): {"Room 3": 0.8, "Outside": 0.2},
        ("Room 2", "Go outside"): {"Outside": 1.0},
        # 
        ("Room 3", "Go outside"): {"Outside": 1.0},
        ("Room 3", "Search"): {"Found item": 0.8, "Outside": 0.2},
        # 
        ("Room 4", "Search"): {"Found item": 0.8, "Outside": 0.2},
        # 
        ("Outside", "Go inside"): {"Room 2": 0.8, "Outside": 0.2}  
        }
    
    return states, reward, transition_prob


def get_policy_random_task_3_and_4():
    """
    Creates a random policy for the service dog example used in "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        policy: The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
    """

    # Policy [𝜋(a|s)]: The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
    policy_random = {
        "Room 1": {"Go to room 1": 1/3, "Go to room 2": 1/3, "Go to room 4": 1/3},
        "Room 2": {"Go to room 3": 0.5, "Go outside": 0.5},
        "Room 3": {"Go outside": 0.5, "Search": 0.5},
        "Room 4": {"Search": 1.0},
        "Outside": {"Go inside": 1.0},
        "Found item": {}
    }

    return policy_random


def get_policy_task_5():
    """
    Creates a random policy for the service dog example used in "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        policy: The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
    """

    # Define random policy [𝜋(a|s)]: The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
    policy = {
        "Room 1": {"Go to room 4": 1.0},
        "Room 2": {"Go to room 3": 0.8, "Go outside": 0.2},
        "Room 3": {"Go outside": 0.5, "Search": 0.5},
        "Room 4": {"Search": 1.0},
        "Outside": {"Go inside": 1.0},
        "Found item": {}
    }

    return policy

