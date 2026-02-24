from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env
import copy


def get_data_textbook():
    """
    Creates the model-free data for the service dog example used in "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
        legal_actions: Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠
    """
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

    # Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠
    legal_actions = {
        "Room 1": ["Go to room 2"],
        "Room 2": ["Go to room 1", "Go to room 3", "Go outside"],
        "Room 3": ["Go to room 2", "Search"],
        "Outside": ["Go inside", "Go outside"],
        "Found item": []
    }

    return legal_actions, reward, transition_prob


class EnvServiceDog():

    def __init__(self, legal_actions, reward, transition_prob):

        # Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠     
        self.legal_actions = legal_actions

        # Reward [R(s,a)]: Reward of taking action 𝑎 in state 𝑠 
        self.reward = reward

        # Transition probability [P(s'|s,a)]: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
        # depends on the current state 𝑠 and the action 𝑎 chosen by the agent. There may be multiple successor states.
        self.transition_prob = transition_prob

        # Dog location = observation = state
        self._dog_location = None   # Will be initialized in self.reset()

        # Random generator
        self.np_random = None # Will be initialized in self.reset()

        # Step counter
        self._step_counter = None # Will be initialized in self.reset()

    def _get_obs(self):
        """Convert internal environment state to observation format.

        Returns:
            Observation = dog location as string
        """

        return str(self._dog_location)
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            Information about the dog's location and the step counter (of the current episode)
        """

        return {"dog location": self._dog_location, "step": self._step_counter}
  
    def reset(self, seed: Optional[int] = None):
        """Start a new episode.

        Returns:
            tuple: (observation, info) for the initial period
        """
        # Reset random generator
        self.np_random = np.random.default_rng(seed=seed)

        # Initial location of the dog
        self._dog_location = "Room 1"

        # Initialize step counter
        self._step_counter = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """The step() method contains the core environment logic. 

        Args:
            action: The action to take

        Returns:
            tuple: (obs, reward, terminated, truncated, info)
        """

        # Get current observation
        obs = self._get_obs()

        # Compute reward
        reward = self.reward[(obs, action)]

        # Update dog's location
        _transition_prob = self.transition_prob[(obs, action)]
        self._dog_location = str(self.np_random.choice(a=list(_transition_prob.keys()), 
                                                            p=list(_transition_prob.values())))

        # Update step counter
        self._step_counter += 1

        # Get observation and info
        obs = self._get_obs()
        info = self._get_info()

        # Check if agent reached the target        
        terminated = True if obs == "Found item" else False

        # Check if episode reached step limit
        truncated = True if self._step_counter >= 500 else False

        return obs, reward, terminated, truncated, info
  
    def _get_legal_actions(self):
        """This method returns a list of legal actions for the current observation. 

        Returns:
            List of legal actions
        """
        
        return self.legal_actions[self._get_obs()]


class EnvServiceDogGym(gym.Env):

    def __init__(self, legal_actions, reward, transition_prob):

        # Initialize from gymansium environment
        super().__init__()
       
        # Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠     
        self.legal_actions = legal_actions

        # Reward [R(s,a)]: Reward of taking action 𝑎 in state 𝑠 
        self.reward = reward

        # Transition probability [P(s'|s,a)]: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
        # depends on the current state 𝑠 and the action 𝑎 chosen by the agent. There may be multiple successor states.
        self.transition_prob = transition_prob

        # Dog location = observation = state
        self._dog_location = None   # Will be initialized in self.reset()

        # Step counter
        self._step_counter = None   # Will be initialized in self.reset()

        # Define what the agent can observe.        
        self.observation_space = gym.spaces.Discrete(len(self.legal_actions.keys()))

        # Map observation numbers to actual observation names to make the code more readable than using raw numbers.
        self._observation_to_room = {id: state for id, state in enumerate(self.legal_actions.keys())}
        self._room_to_observation = {state: id for id, state in self._observation_to_room.items()}  # Create reverse mapping

        # Define what actions are available (action space)
        all_actions = {a for actions in self.legal_actions.values() for a in actions}
        self.action_space = gym.spaces.Discrete(len(all_actions))

        # Map action numbers to actual movements to make the code more readable than using raw numbers.
        self._action_to_direction = {id: action for id, action in enumerate(all_actions)}        
        self._direction_to_action = {action: id for id, action in self._action_to_direction.items()}    # Create reverse mapping
        
    def _get_obs(self):
        """Convert internal environment state to observation format.

        Returns:
            Observation = dog location as integer
        """

        return self._dog_location
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            Information about the dog's location and the step counter (of the current episode)
        """

        return {"dog location": self._dog_location, "step": self._step_counter}
  
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Returns:
            tuple: (observation, info) for the initial period
        """
        # IMPORTANT: Must call this first to seed the random number generator (called via self.np_random)
        super().reset(seed=seed)

        # Initial location of the dog
        self._dog_location = self._room_to_observation["Room 1"]

        # Initialize step counter
        self._step_counter = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """The step() method contains the core environment logic. 

        Args:
            action: The action to take as string

        Returns:
            tuple: (obs, reward, terminated, truncated, info)
        """

        # Get current observation and action as readable values
        obs = self._observation_to_room[self._get_obs()]
        if isinstance(action, (int, np.integer)):
            action = self._action_to_direction[action]

        # Compute reward
        reward = self.reward.get((obs, action), 0)

        # Update dog's location
        _transition_prob = self.transition_prob.get((obs, action), {"Found item": 1.0})
        
        self._dog_location = self._room_to_observation[str(self.np_random.choice(
            a=list(_transition_prob.keys()), 
            p=list(_transition_prob.values())
            ))]

        # Update step counter
        self._step_counter += 1

        # Get observation and info
        obs = self._get_obs()
        info = self._get_info()
        
        # Check if agent reached the target        
        terminated = True if self._observation_to_room[obs] == "Found item" else False

        # Check if episode reached step limit
        truncated = True if self._step_counter >= 500 else False

        return obs, reward, terminated, truncated, info
  
    def _get_legal_actions(self):
        """This method returns a list of legal actions for the current observation. 

        Returns:
            List of legal actions as strings
        """

        return self.legal_actions[self._observation_to_room[self._get_obs()]]

    def check(self):
        """This method catches many common issues with the Gymnasium environment

        """

        try:
            check_env(self)
            print("Environment passes all checks!")
        except Exception as e:
            print(f"Environment has issues: {e}")


class PolicyRandom():
    
    def __init__(self):       
        pass
    
    def _get_action(self, env):
        """Returns an action based on a policy in which legal actions are selected with equal probability. 

        Args:
            env: The environment action is taken in

        Returns:
            A legal action 
        """

        # Get legal actions for current period of the environment
        legal_actions = env._get_legal_actions()

        # Randomly select one of the legal actions (equal probability)
        action = str(env.np_random.choice(legal_actions))

        return action
        
    
class PolicySARSA():

    def __init__(self):
        
        # Intialize lookup table for state-action values (Q-table)
        self.Q = dict()

    def _get_action(self, env, epsilon=0):
        """Returns an action based on and epsilon-greedy policy.  

        Args:
            env: The environment action is taken in
            epsilon: The probability of a taking a random action. If epsilon=0, the policy behaves greedy.

        Returns:
            A legal action 
        """
        
        # Get legal actions for current period of the environment
        legal_actions = env._get_legal_actions()
        p = env.np_random.random()
        
        if epsilon > 0 and p < epsilon:
            # Draw action among all legal actions with equal probability
            action = str(env.np_random.choice(legal_actions))
        
        else:
            # Get state
            state = env._get_obs()
            
            # Initialize Q values if not available
            if state not in self.Q:
                self.Q[state] = {}
            for a in legal_actions:
                if a not in self.Q[state]:
                    self.Q[state][a] = 0

            # Select action with the highest value             
            action = max(self.Q[state], key=self.Q[state].get)
        
        # print(f"p={p} -> action={action}")

        return action

    def _learn(self, env, discount, epsilon, learning_rate, num_updates, seed: Optional[int] = None):
         """Learning of the (optimal) state-action values (=Q values) via on-policy SARSA. 
           The learned values are stored in the dictionary self.Q

        Args:
            env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
            discount: discount factor, must be 0 <= discount <= 1.
            epsilon: exploration rate for the e-greedy policy, must be 0 <= epsilon < 1.
            learning_rate: the learning rate when update step size
            num_updates: number of updates to the value function. Must be integer. 
            on_policy: =True if behavior policy and target policy are the same (SARSA), =False if target policy is greedy (Q-Learning)
            seed: The seed for the learning phase (this will be passed on the the environment)

        """
         
         td_learning(policy=self, env=env, discount=discount, epsilon=epsilon, learning_rate=learning_rate, 
                    num_updates=num_updates, seed=seed, on_policy=True)
       

class PolicyQLearning():

    def __init__(self):
        
        # Intialize lookup table for state-action values (Q-table)
        self.Q = dict()

    def _get_action(self, env, epsilon=0):
        """Returns an action based on and epsilon-greedy policy.  

        Args:
            env: The environment action is taken in
            epsilon: The probability of a taking a random action. If epsilon=0, the policy behaves greedy.

        Returns:
            A legal action 
        """
        
        # Get legal actions for current period of the environment
        legal_actions = env._get_legal_actions()

        if epsilon > 0 and env.np_random.random() < epsilon:
            # Draw action among all legal actions with equal probability
            action = str(env.np_random.choice(legal_actions))
        else:
            # Get state
            state = env._get_obs()
            
            # Initialize Q values if not available
            if state not in self.Q:
                self.Q[state] = {}
            for a in legal_actions:
                if a not in self.Q[state]:
                    self.Q[state][a] = 0

            # Select action with the highest value             
            action = max(self.Q[state], key=self.Q[state].get)
        
        return action

    def _learn(self, env, discount, epsilon, learning_rate, num_updates, seed: Optional[int] = None):
         """Learning of the (optimal) state-action values (=Q values) via off-policy Q-learning. 
           The learned values are stored in the dictionary self.Q

        Args:
            env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
            discount: discount factor, must be 0 <= discount <= 1.
            epsilon: exploration rate for the e-greedy policy, must be 0 <= epsilon < 1.
            learning_rate: the learning rate when update step size
            num_updates: number of updates to the value function. Must be integer. 
            on_policy: =True if behavior policy and target policy are the same (SARSA), =False if target policy is greedy (Q-Learning)
            seed: The seed for the learning phase (this will be passed on the the environment)

        """
         
         td_learning(policy=self, env=env, discount=discount, epsilon=epsilon, learning_rate=learning_rate, 
                    num_updates=num_updates, seed=seed, on_policy=False)
       

def compute_returns(rewards, discount):
    """Compute returns for every time step in the episode trajectory.

    Args:
        rewards: a list of rewards from an episode.
        discount: discount factor, must be 0 <= discount <= 1.

    Returns:
        returns: return for every single time step in the episode trajectory.
    """
    assert 0.0 <= discount <= 1.0

    returns = []
    G_t = 0
    # We do it backwards so it's more efficient and easier to implement.
    for t in reversed(range(len(rewards))):
        G_t = rewards[t] + discount * G_t
        returns.append(G_t)
    returns.reverse()

    return returns


def mc_policy_evaluation(env, policy, discount, num_episodes, first_visit=True, seed: Optional[int] = None):
    """Run Monte Carlo policy evaluation for state value function.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        policy: the policy that we want to evaluate.
        discount: discount factor, must be 0 <= discount <= 1.
        num_episodes: number of episodes to run.
        first_visit: use first-visit MC, default on.

    Returns:
        V: the estimated state value function for the input policy after run evaluation for num_episodes.
    """
    assert 0.0 <= discount <= 1.0
    assert isinstance(num_episodes, int)

    # Initialize
    N = dict()  # counter for visits number
    V = dict()  # state value function
    G = dict()  # total returns

    for _ in range(num_episodes):
        # Sample an episode trajectory using the given policy.
        episode = []
        state, info = env.reset(seed=seed)
        seed = seed if seed is None else seed + 1

        while True:
            # Get action when following the policy.
            action = policy._get_action(env)

            # Take the action in the environment and observe successor state and reward.
            state_tp1, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
            state = state_tp1
            if terminated or truncated:
                break

        # Unpack list of tuples into separate lists.
        # print(episode)
        states, _, rewards = map(list, zip(*episode))

        # Compute returns for every time step in the episode.
        returns = compute_returns(rewards, discount)

        # Loop over all state in the episode.
        for t, state in enumerate(states):

            G_t = returns[t]
            # Check if this is the first time state visited in the episode.
            if first_visit and state in states[:t]:
                continue

            N[state] = N.get(state, 0) + 1
            G[state] = G.get(state, 0) + G_t
            V[state] = G[state] / N[state]

    # Round state value function
    for state in V:
        V[state] = round(V[state], 2)

    print(f"MC Policy evaluation results of state value function after {num_episodes} iterations: {V}")


def td_learning(policy, env, discount, epsilon, learning_rate, num_updates, on_policy: Optional[bool] = True, seed: Optional[int] = None):
    """Learning of the (optimal) state-action values (=Q values). 
        The learned values are stored in the dictionary policy.Q

    Args:
        policy: a policy class in which the state-action values are stored and in which the decision making is described
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        discount: discount factor, must be 0 <= discount <= 1.
        epsilon: exploration rate for the e-greedy policy, must be 0 <= epsilon < 1.
        learning_rate: the learning rate when update step size
        num_updates: number of updates to the value function. Must be integer. 
        on_policy: =True if behavior policy and target policy are the same (SARSA), =False if target policy is greedy (Q-Learning)
        seed: The seed for the learning phase (this will be passed on the the environment)

    """

    # Initialize state-action value function
    policy.Q = dict()

    # Initialize conuter for number of updates
    i = 0

    # Reset environment
    state, info = env.reset(seed=seed)

    # Get initial action by following the policy.
    action = policy._get_action(env, epsilon)

    while i < num_updates:

        # Take the action in the environment and observe successor state and reward.
        state_tp1, reward, terminated, truncated, info = env.step(action)

        # Compute TD target (first part, immediate reward)       
        delta = reward

        if not terminated and not truncated:
            # TP 1 action: On-policy = SARSA, Off-Policy = Q-Learning
            action_tp1 = policy._get_action(env, epsilon) if on_policy else policy._get_action(env, epsilon=0)

            # Compute TD target (second part, downstream reward)
            policy.Q[state_tp1] = policy.Q.get(state_tp1, {})
            policy.Q[state_tp1][action_tp1] = policy.Q[state_tp1].get(action_tp1, 0)
            delta += discount * policy.Q[state_tp1][action_tp1]

        # Update Q value
        policy.Q[state] = policy.Q.get(state, {})
        policy.Q[state][action] = policy.Q[state].get(action, 0) + learning_rate * (delta - policy.Q[state].get(action, 0))

        if terminated or truncated:
            # Update seed
            seed = seed if seed is None else seed + 1

            # Reset environment
            state, info = env.reset(seed=seed)

            # Get initial action by following the policy.
            action = policy._get_action(env, epsilon)

        else:
            # Update state
            state = state_tp1
            
            if on_policy:
                # Get action via TD1 action (as this was selected via the same policy).
                action = copy.deepcopy(action_tp1)
            else: 
                # Get action by following the policy.
                action = policy._get_action(env, epsilon)

        # Update counter
        i += 1
        # print(f"Q after {i} iterations: {policy.Q}")


if __name__ == "__main__":

    # Load data
    legal_actions, reward, transition_prob = get_data_textbook()

    # Create environment
    env = EnvServiceDog(legal_actions, reward, transition_prob)
    # env = EnvServiceDogGym(legal_actions, reward, transition_prob)
    
    # Random policy
    print("RANDOM POLICY")
    policy = PolicyRandom()
    mc_policy_evaluation(env, policy, discount=0.9, num_episodes=1000, first_visit=True, seed=2506)
    
    # SARSA
    # print("TD SARSA")
    # policy = PolicySARSA()
    # policy._learn(env, discount=0.9, epsilon=0.4, learning_rate=0.01, num_updates=500, seed=2506)
    # print(policy.Q)
    # mc_policy_evaluation(env, policy, discount=0.9, num_episodes=1000, first_visit=True, seed=2506)

    # Q-Learning
    # print("TD Q-Learning")
    # policy = PolicyQLearning()
    # policy._learn(env, discount=0.9, epsilon=0.4, learning_rate=0.01, num_updates=20000, seed=2506)
    # mc_policy_evaluation(env, policy, discount=0.9, num_episodes=1000, first_visit=True, seed=2506)
