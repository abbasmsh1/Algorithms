import numpy as np
import random
from collections import defaultdict, deque

class QLearning:
    """
    Q-Learning algorithm for reinforcement learning.
    
    Q-Learning is a model-free reinforcement learning algorithm that learns a
    policy telling an agent what action to take under what circumstances.
    
    Attributes:
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Rate at which to decay epsilon.
        min_epsilon (float): Minimum value for epsilon.
        q_table (defaultdict): Mapping from (state, action) pairs to Q-values.
    """
    
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Initialize the Q-Learning agent.
        
        Args:
            alpha (float): Learning rate. Default is 0.1.
            gamma (float): Discount factor. Default is 0.99.
            epsilon (float): Exploration rate. Default is 1.0.
            epsilon_decay (float): Rate at which to decay epsilon. Default is 0.995.
            min_epsilon (float): Minimum value for epsilon. Default is 0.01.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))  # Default to 4 actions
    
    def get_action(self, state, num_actions=None):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: The current state.
            num_actions (int, optional): Number of possible actions. Default is None.
            
        Returns:
            int: The chosen action.
        """
        # If num_actions is specified, override the default
        n_actions = num_actions if num_actions is not None else len(self.q_table[state])
        
        # Exploration: choose a random action
        if random.random() < self.epsilon:
            return random.randint(0, n_actions - 1)
        
        # Exploitation: choose the best action
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-value for the state-action pair.
        
        Args:
            state: The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state: The next state.
            done (bool): Whether the episode has terminated.
            
        Returns:
            float: The updated Q-value.
        """
        # Calculate the best Q-value for the next state
        best_next_action = np.argmax(self.q_table[next_state])
        best_next_value = self.q_table[next_state][best_next_action]
        
        # If the episode is done, there is no next state value
        if done:
            best_next_value = 0
        
        # Compute the TD target
        td_target = reward + self.gamma * best_next_value
        
        # Compute the TD error
        td_error = td_target - self.q_table[state][action]
        
        # Update the Q-value
        self.q_table[state][action] += self.alpha * td_error
        
        return self.q_table[state][action]
    
    def decay_epsilon(self):
        """
        Decay the exploration rate.
        
        Returns:
            float: The new epsilon value.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon


class SARSA:
    """
    SARSA (State-Action-Reward-State-Action) algorithm for reinforcement learning.
    
    SARSA is an on-policy TD control algorithm that learns the Q-values for
    state-action pairs while following the current policy.
    
    Attributes:
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Rate at which to decay epsilon.
        min_epsilon (float): Minimum value for epsilon.
        q_table (defaultdict): Mapping from (state, action) pairs to Q-values.
    """
    
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Initialize the SARSA agent.
        
        Args:
            alpha (float): Learning rate. Default is 0.1.
            gamma (float): Discount factor. Default is 0.99.
            epsilon (float): Exploration rate. Default is 1.0.
            epsilon_decay (float): Rate at which to decay epsilon. Default is 0.995.
            min_epsilon (float): Minimum value for epsilon. Default is 0.01.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))  # Default to 4 actions
    
    def get_action(self, state, num_actions=None):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: The current state.
            num_actions (int, optional): Number of possible actions. Default is None.
            
        Returns:
            int: The chosen action.
        """
        # If num_actions is specified, override the default
        n_actions = num_actions if num_actions is not None else len(self.q_table[state])
        
        # Exploration: choose a random action
        if random.random() < self.epsilon:
            return random.randint(0, n_actions - 1)
        
        # Exploitation: choose the best action
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        """
        Update the Q-value for the state-action pair using SARSA.
        
        Args:
            state: The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state: The next state.
            next_action (int): The next action to be taken.
            done (bool): Whether the episode has terminated.
            
        Returns:
            float: The updated Q-value.
        """
        # If the episode is done, there is no next state value
        next_value = 0 if done else self.q_table[next_state][next_action]
        
        # Compute the TD target
        td_target = reward + self.gamma * next_value
        
        # Compute the TD error
        td_error = td_target - self.q_table[state][action]
        
        # Update the Q-value
        self.q_table[state][action] += self.alpha * td_error
        
        return self.q_table[state][action]
    
    def decay_epsilon(self):
        """
        Decay the exploration rate.
        
        Returns:
            float: The new epsilon value.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon


class ExpectedSARSA:
    """
    Expected SARSA algorithm for reinforcement learning.
    
    Expected SARSA is a variation of SARSA that uses the expected value of the
    next state-action pair instead of a sample.
    
    Attributes:
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Rate at which to decay epsilon.
        min_epsilon (float): Minimum value for epsilon.
        q_table (defaultdict): Mapping from (state, action) pairs to Q-values.
    """
    
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Initialize the Expected SARSA agent.
        
        Args:
            alpha (float): Learning rate. Default is 0.1.
            gamma (float): Discount factor. Default is 0.99.
            epsilon (float): Exploration rate. Default is 1.0.
            epsilon_decay (float): Rate at which to decay epsilon. Default is 0.995.
            min_epsilon (float): Minimum value for epsilon. Default is 0.01.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))  # Default to 4 actions
    
    def get_action(self, state, num_actions=None):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: The current state.
            num_actions (int, optional): Number of possible actions. Default is None.
            
        Returns:
            int: The chosen action.
        """
        # If num_actions is specified, override the default
        n_actions = num_actions if num_actions is not None else len(self.q_table[state])
        
        # Exploration: choose a random action
        if random.random() < self.epsilon:
            return random.randint(0, n_actions - 1)
        
        # Exploitation: choose the best action
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done, num_actions=None):
        """
        Update the Q-value for the state-action pair using Expected SARSA.
        
        Args:
            state: The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state: The next state.
            done (bool): Whether the episode has terminated.
            num_actions (int, optional): Number of possible actions. Default is None.
            
        Returns:
            float: The updated Q-value.
        """
        # If num_actions is specified, override the default
        n_actions = num_actions if num_actions is not None else len(self.q_table[next_state])
        
        # If the episode is done, there is no next state value
        if done:
            expected_value = 0
        else:
            # Calculate the expected value using the epsilon-greedy policy
            best_next_action = np.argmax(self.q_table[next_state])
            
            # Probability of selecting best action
            prob_best = 1 - self.epsilon + self.epsilon / n_actions
            
            # Probability of selecting other actions
            prob_other = self.epsilon / n_actions
            
            # Calculate expected value
            expected_value = 0
            for a in range(n_actions):
                if a == best_next_action:
                    expected_value += prob_best * self.q_table[next_state][a]
                else:
                    expected_value += prob_other * self.q_table[next_state][a]
        
        # Compute the TD target
        td_target = reward + self.gamma * expected_value
        
        # Compute the TD error
        td_error = td_target - self.q_table[state][action]
        
        # Update the Q-value
        self.q_table[state][action] += self.alpha * td_error
        
        return self.q_table[state][action]
    
    def decay_epsilon(self):
        """
        Decay the exploration rate.
        
        Returns:
            float: The new epsilon value.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon


class MonteCarloControl:
    """
    Monte Carlo Control with Exploring Starts for reinforcement learning.
    
    This algorithm uses Monte Carlo methods to learn the optimal policy
    by estimating action values from complete episodes.
    
    Attributes:
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Rate at which to decay epsilon.
        min_epsilon (float): Minimum value for epsilon.
        q_table (defaultdict): Mapping from (state, action) pairs to Q-values.
        returns (defaultdict): Mapping from (state, action) pairs to returns.
        returns_count (defaultdict): Count of returns for each (state, action) pair.
    """
    
    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Initialize the Monte Carlo Control agent.
        
        Args:
            gamma (float): Discount factor. Default is 0.99.
            epsilon (float): Exploration rate. Default is 1.0.
            epsilon_decay (float): Rate at which to decay epsilon. Default is 0.995.
            min_epsilon (float): Minimum value for epsilon. Default is 0.01.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))  # Default to 4 actions
        self.returns = defaultdict(list)  # List of returns for each state-action pair
        self.returns_count = defaultdict(int)  # Count of returns for each state-action pair
    
    def get_action(self, state, num_actions=None):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: The current state.
            num_actions (int, optional): Number of possible actions. Default is None.
            
        Returns:
            int: The chosen action.
        """
        # If num_actions is specified, override the default
        n_actions = num_actions if num_actions is not None else len(self.q_table[state])
        
        # Exploration: choose a random action
        if random.random() < self.epsilon:
            return random.randint(0, n_actions - 1)
        
        # Exploitation: choose the best action
        return np.argmax(self.q_table[state])
    
    def update_from_episode(self, episode):
        """
        Update Q-values from a complete episode.
        
        Args:
            episode (list): List of (state, action, reward) tuples from an episode.
            
        Returns:
            None
        """
        states, actions, rewards = zip(*episode)
        
        # Calculate discounted returns for each step
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # Update Q-values using first-visit Monte Carlo
        state_action_pairs = set()
        for i, (state, action, _) in enumerate(episode):
            state_action = (state, action)
            
            # Only update if this is the first occurrence of the state-action pair
            if state_action not in state_action_pairs:
                state_action_pairs.add(state_action)
                
                # Add the return to the list of returns
                self.returns[(state, action)].append(returns[i])
                
                # Update the Q-value as the average of all returns
                self.q_table[state][action] = np.mean(self.returns[(state, action)])
    
    def decay_epsilon(self):
        """
        Decay the exploration rate.
        
        Returns:
            float: The new epsilon value.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    from collections import deque
    import random
    
    class DQN:
        """
        Deep Q-Network for reinforcement learning.
        
        DQN uses a neural network to approximate the Q-function, allowing it to
        handle high-dimensional state spaces.
        
        Attributes:
            state_size (int): Dimension of the state space.
            action_size (int): Number of possible actions.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
            epsilon_decay (float): Rate at which to decay epsilon.
            min_epsilon (float): Minimum value for epsilon.
            learning_rate (float): Learning rate for the optimizer.
            batch_size (int): Size of batches for training.
            memory (deque): Replay memory to store experiences.
            model (nn.Module): Q-network model.
            optimizer (torch.optim.Optimizer): Optimizer for the model.
        """
        
        class QNetwork(nn.Module):
            """
            Neural network for approximating the Q-function.
            """
            
            def __init__(self, state_size, action_size, hidden_size=64):
                """
                Initialize the Q-network.
                
                Args:
                    state_size (int): Dimension of the state space.
                    action_size (int): Number of possible actions.
                    hidden_size (int): Size of hidden layers. Default is 64.
                """
                super(DQN.QNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, action_size)
            
            def forward(self, x):
                """
                Forward pass through the network.
                
                Args:
                    x (torch.Tensor): Input tensor.
                    
                Returns:
                    torch.Tensor: Output tensor.
                """
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        
        def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0,
                    epsilon_decay=0.995, min_epsilon=0.01, learning_rate=0.001,
                    batch_size=64, memory_size=10000):
            """
            Initialize the DQN agent.
            
            Args:
                state_size (int): Dimension of the state space.
                action_size (int): Number of possible actions.
                gamma (float): Discount factor. Default is 0.99.
                epsilon (float): Exploration rate. Default is 1.0.
                epsilon_decay (float): Rate at which to decay epsilon. Default is 0.995.
                min_epsilon (float): Minimum value for epsilon. Default is 0.01.
                learning_rate (float): Learning rate for the optimizer. Default is 0.001.
                batch_size (int): Size of batches for training. Default is 64.
                memory_size (int): Size of replay memory. Default is 10000.
            """
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.min_epsilon = min_epsilon
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            
            # Initialize replay memory
            self.memory = deque(maxlen=memory_size)
            
            # Initialize model and target network
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.QNetwork(state_size, action_size).to(self.device)
            self.target_model = self.QNetwork(state_size, action_size).to(self.device)
            self.update_target_model()
            
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        def update_target_model(self):
            """
            Update the target network's weights with the current model's weights.
            """
            self.target_model.load_state_dict(self.model.state_dict())
        
        def remember(self, state, action, reward, next_state, done):
            """
            Add experience to memory.
            
            Args:
                state: The current state.
                action (int): The action taken.
                reward (float): The reward received.
                next_state: The next state.
                done (bool): Whether the episode has terminated.
            """
            self.memory.append((state, action, reward, next_state, done))
        
        def get_action(self, state):
            """
            Choose an action using epsilon-greedy policy.
            
            Args:
                state: The current state.
                
            Returns:
                int: The chosen action.
            """
            # Convert state to tensor
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Exploration: choose a random action
            if random.random() < self.epsilon:
                return random.randint(0, self.action_size - 1)
            
            # Exploitation: choose the best action
            with torch.no_grad():
                q_values = self.model(state)
                return torch.argmax(q_values).item()
        
        def replay(self):
            """
            Train the model on a batch of experiences from memory.
            
            Returns:
                float: The loss value.
            """
            if len(self.memory) < self.batch_size:
                return 0.0
            
            # Sample a batch of experiences
            batch = random.sample(self.memory, self.batch_size)
            
            # Convert to tensors
            states = torch.FloatTensor([experience[0] for experience in batch]).to(self.device)
            actions = torch.LongTensor([[experience[1]] for experience in batch]).to(self.device)
            rewards = torch.FloatTensor([[experience[2]] for experience in batch]).to(self.device)
            next_states = torch.FloatTensor([experience[3] for experience in batch]).to(self.device)
            dones = torch.FloatTensor([[experience[4]] for experience in batch]).to(self.device)
            
            # Compute Q values
            current_q_values = self.model(states).gather(1, actions)
            
            # Compute target Q values
            with torch.no_grad():
                max_next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            
            # Compute loss
            loss = F.mse_loss(current_q_values, target_q_values)
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        
        def decay_epsilon(self):
            """
            Decay the exploration rate.
            
            Returns:
                float: The new epsilon value.
            """
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            return self.epsilon
        
        def save(self, path):
            """
            Save the model's weights to a file.
            
            Args:
                path (str): File path to save the model.
            """
            torch.save(self.model.state_dict(), path)
        
        def load(self, path):
            """
            Load the model's weights from a file.
            
            Args:
                path (str): File path to load the model from.
            """
            self.model.load_state_dict(torch.load(path))
            self.target_model.load_state_dict(torch.load(path))

except ImportError:
    class DQN:
        """
        Placeholder for Deep Q-Network (requires PyTorch).
        
        This class is a placeholder for when PyTorch is not available.
        """
        
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DQN. Please install it with: pip install torch") 