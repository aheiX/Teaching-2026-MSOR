
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from typing import Optional


class EnvTicTacToe(gym.Env):

    def __init__(self):
        # Initialize from gymansium environment
        super().__init__()

        # Define what the agent can observe (state space). 
        # Every observation/state is a (filled) 3x3 board (9 positions in total) and information about the turn
        self.observation_space = gym.spaces.Dict(
            {"board": gym.spaces.MultiDiscrete([3,3,3, 3,3,3, 3,3,3]),   # Empty (0), X (1), or O (2)
             "turn": gym.spaces.Discrete(n=2, start=1)                  # 1=X, 2=O  
             }
        )

        # Dictionaries to make number-based information readable
        self.pos_value_to_name = {0: "-", 1: "X", 2: "O"}
        self.pos_id_to_name = {0: "top-left", 1: "top-center", 2: "top-right",
                               3: "middle-left", 4: "center", 5: "middle-right",
                               6: "bottom-left", 7: "bottom-center", 8: "bottom-right"}

        # Define what actions are available (action space). 
        # It is the position id (0 to 8) at which the player positions the mark
        self.action_space = gym.spaces.Discrete(9)

    def _get_obs(self):
        """Convert internal state to observation format. 

        Returns:
            Observatio
        """

        return {"board": self.board, "turn": self.turn}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            Information about the board, turn, empty positions, and counter of the moves
        """

        return {"board": self.board, 
                "turn": self.turn, 
                "empty positions": self._get_legal_actions(), 
                "count moves": self.count_moves
                }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Returns:
            tuple: (observation, info) for the initial period
        """
        # IMPORTANT: Must call this first to seed the random number generator (called via self.np_random)
        super().reset(seed=seed)

        # Initialize board (all empty) and turn (player 1 (X) always starts)
        self.board = [0,0,0, 0,0,0, 0,0,0]
        self.turn = 1

        # Reset moves counter
        self.count_moves = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """The step() method contains the core environment logic. 
           It takes an action, updates the environment state, and returns the results. 
           
        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Update moves counter
        self.count_moves += 1

        # Update board
        self.board[action] = self.turn

        # Reward: A small penalty to encourage efficiency. 
        # Notice that the reward "at the end of the game" is not returned by the environment. 
        reward = -0.01  

        # Check for end of the game (has a winner or no empty spots = draw)
        terminated = True if self.is_winner(mark=self.turn) or len(self._get_legal_actions()) == 0 else False
      
        # A step limit is not used in this game
        truncated = False

        # Update turn
        self.turn = 2 if self.turn == 1 else 1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def is_winner(self, mark):
        """This method checks if the mark (1=X, 2=O) has won the game.
           
        Args:
            mark: The mark/symbol for which a winning sequence is checked for

        Returns:
            boolean: True/False
        """

        # Horizontal
        if all(self.board[pos] == mark for pos in [0, 1, 2]):
            return True
        elif all(self.board[pos] == mark for pos in [3, 4, 5]):
            return True
        elif all(self.board[pos] == mark for pos in [6, 7, 8]):
            return True
        
        # Vertical
        elif all(self.board[pos] == mark for pos in [0, 3, 6]):
            return True
        elif all(self.board[pos] == mark for pos in [1, 4, 7]):
            return True
        elif all(self.board[pos] == mark for pos in [2, 5, 8]):
            return True
        
        # Diagonal
        elif all(self.board[pos] == mark for pos in [0, 4, 8]):
            return True
        elif all(self.board[pos] == mark for pos in [2, 4, 6]):
            return True
        
        return False

    def _get_legal_actions(self):
        """This method returns all legal actions of the current observation. 
           
        Returns:
            list: list of empty positions
        """

        
        board = self._get_obs()["board"]

        return [pos for pos, val in enumerate(board) if val == 0]

    def print_current_board(self):
        """This method prints the current board. 
           
        """

        print(f"Board after {self.count_moves} moves:")

        readible_board = [self.pos_value_to_name[pos] for pos in self._get_obs()["board"]]

        print(readible_board[0:3], '[0,1,2]')
        print(readible_board[3:6], '[3,4,5]')
        print(readible_board[6:9], '[6,7,8]')

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
        action = env.np_random.choice(legal_actions)

        return action


def play(env, opponents_policy=None):
    """
    Enables a game of Tic-Tac-Toe where the user plays as Player X via console input.
    The opponent's policy can be either human (console input) or an automated policy.
    
    Args:
        env: The game environment that follows a standard gymnsasium interface with reset(), step(), and print_current_board() methods.
        opponents_policy: Optional; an object with a get_action() method that determines the opponent's moves automatically.
                          If None, the opponent will be controlled via console input.
    """
    keep_playing = True  # Control variable for multiple game sessions
    
    while keep_playing:
        # Reset environment to start a new game episode
        observation, info = env.reset()

        # Flag to determine if the current game has ended
        episode_over = False  

        while not episode_over:
            # Display the current state of the game board
            env.print_current_board()

            # Prompt the user (Player X) for a move
            action = int(input(f"Your (valid) move (cell number) as player {env.pos_value_to_name[env.turn]}: "))

            # Apply the player's move to the environment
            observation, reward, terminated, truncated, info = env.step(action)
            # Determine if the game has ended either by victory, loss, or draw
            episode_over = terminated or truncated

            if not episode_over and opponents_policy is not None:
                # Automated opponent's turn based on the supplied policy
                
                # Select opponent's move using the policy's get_action method
                action = opponents_policy._get_action(env)
                print(f"Other player's ({env.pos_value_to_name[env.turn]}) move: {action} ({env.pos_id_to_name[action]})")
                
                # Debug: Show Q-values for each legal action (optional)
                try:
                    legal_actions = env._get_legal_actions()                
                    action_info = {
                        env.pos_id_to_name[a]: f"{round(opponents_policy.Q[tuple(observation['board'])][a], 2)}"
                        for a in legal_actions
                    }
                    print(f"{action_info}")
                except:
                    pass

                # Execute opponent's move
                observation, reward, terminated, truncated, info = env.step(action)

                # Check if game has ended after opponent's move
                episode_over = terminated or truncated

        # After the game ends, display final board
        env.print_current_board()

        # Determine and display the game result
        if env.is_winner(mark=1):
            result = "X wins!"
        elif env.is_winner(mark=2):
            result = "O wins!"
        else:
            result = "Draw"
        print(f"=> Result: {result}")

        # Close the environment
        env.close()

        # Ask user whether to continue playing
        user_input = input("Continue playing (y=yes, n=no): ").strip().lower()
        keep_playing = True if user_input == "y" else False


if __name__ == "__main__":
    # Check environment
    # env=EnvTicTacToe()
    # env.check()
    
    # Play against each other (both via console input)
    play(env=EnvTicTacToe(), opponents_policy=None)

    # Play against a computer following a random strategy. You should win every game. 
    # policy = PolicyRandom()     
    # play(env=EnvTicTacToe(), opponents_policy=policy)
