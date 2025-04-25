import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque


class EpsilonScheduler:
    """
    Scheduler for the exploration rate (epsilon) in reinforcement learning.
    
    Supports different decay strategies for epsilon to balance
    exploration and exploitation during training.
    
    Attributes:
        strategy (str): Strategy for epsilon decay ('linear', 'exponential', or 'constant').
        start_value (float): Initial value of epsilon.
        end_value (float): Final value of epsilon.
        decay_steps (int): Number of steps to decay epsilon over.
        current_step (int): Current step in the decay process.
    """
    
    def __init__(self, strategy='exponential', start_value=1.0, end_value=0.01, decay_steps=10000):
        """
        Initialize the epsilon scheduler.
        
        Args:
            strategy (str): Strategy for epsilon decay. Options are 'linear', 'exponential', or 'constant'.
                Default is 'exponential'.
            start_value (float): Initial value of epsilon. Default is 1.0.
            end_value (float): Final value of epsilon. Default is 0.01.
            decay_steps (int): Number of steps to decay epsilon over. Default is 10000.
        """
        self.strategy = strategy
        self.start_value = start_value
        self.end_value = end_value
        self.decay_steps = decay_steps
        self.current_step = 0
        
        # Calculate decay rate for exponential decay
        if strategy == 'exponential':
            self.decay_rate = np.power(end_value / start_value, 1.0 / decay_steps)
    
    def get_value(self):
        """
        Get the current epsilon value.
        
        Returns:
            float: Current epsilon value.
        """
        if self.strategy == 'constant':
            return self.start_value
        
        elif self.strategy == 'linear':
            fraction = min(1.0, self.current_step / self.decay_steps)
            return self.start_value + fraction * (self.end_value - self.start_value)
        
        elif self.strategy == 'exponential':
            return max(self.end_value, self.start_value * (self.decay_rate ** self.current_step))
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def step(self):
        """
        Increment the step counter and return the new epsilon value.
        
        Returns:
            float: Updated epsilon value.
        """
        self.current_step += 1
        return self.get_value()


class ExperienceReplay:
    """
    Memory buffer for storing and sampling experiences in reinforcement learning.
    
    This implementation uses a deque with a fixed maximum length to store
    experiences and provides methods to add experiences and sample batches.
    
    Attributes:
        memory (deque): Buffer for storing experiences.
        batch_size (int): Size of batches to sample.
    """
    
    def __init__(self, capacity=10000, batch_size=64):
        """
        Initialize the experience replay buffer.
        
        Args:
            capacity (int): Maximum number of experiences to store. Default is 10000.
            batch_size (int): Size of batches to sample. Default is 64.
        """
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode has terminated.
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self):
        """
        Sample a batch of experiences from the buffer.
        
        Returns:
            tuple: Batch of states, actions, rewards, next_states, and dones.
        """
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            int: Current number of experiences in the buffer.
        """
        return len(self.memory)


def plot_learning_curve(rewards, window=100, title='Learning Curve'):
    """
    Plot the rewards per episode with a moving average to visualize learning progress.
    
    Args:
        rewards (list): List of rewards per episode.
        window (int): Size of the moving average window. Default is 100.
        title (str): Title for the plot. Default is 'Learning Curve'.
    
    Returns:
        tuple: Figure and axes objects.
    """
    # Calculate the moving average
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the raw rewards
    ax.plot(rewards, alpha=0.6, label='Raw Rewards')
    
    # Plot the moving average
    ax.plot(np.arange(window-1, len(rewards)), moving_avg, 
            label=f'Moving Average (window={window})')
    
    # Add labels and legend
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig, ax


def plot_state_values(value_function, width, height, title='State Values'):
    """
    Plot the state values for a grid world environment.
    
    Args:
        value_function (dict or function): Mapping from states to values.
        width (int): Width of the grid.
        height (int): Height of the grid.
        title (str): Title for the plot. Default is 'State Values'.
    
    Returns:
        tuple: Figure and axes objects.
    """
    # Create a grid of values
    values = np.zeros((height, width))
    
    # Fill in the values
    for y in range(height):
        for x in range(width):
            state = (x, y)
            
            # Check if value_function is a dict or a function
            if callable(value_function):
                values[height - 1 - y, x] = value_function(state)
            else:
                values[height - 1 - y, x] = value_function.get(state, 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the values as a heatmap
    im = ax.imshow(values, cmap='viridis')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Value')
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add grid
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    # Add the values as text
    for y in range(height):
        for x in range(width):
            ax.text(x, height - 1 - y, f"{values[height - 1 - y, x]:.2f}",
                   ha='center', va='center', color='white' if values[height - 1 - y, x] < 0.5 else 'black')
    
    return fig, ax


def plot_policy(policy, width, height, title='Policy'):
    """
    Plot the policy for a grid world environment.
    
    Args:
        policy (dict or function): Mapping from states to actions.
        width (int): Width of the grid.
        height (int): Height of the grid.
        title (str): Title for the plot. Default is 'Policy'.
    
    Returns:
        tuple: Figure and axes objects.
    """
    # Create a grid for the policy directions
    U = np.zeros((height, width))  # x component of the arrow
    V = np.zeros((height, width))  # y component of the arrow
    
    # Action to direction mapping
    # Assuming actions are: 0 = up, 1 = right, 2 = down, 3 = left
    directions = {
        0: (0, 1),   # up
        1: (1, 0),   # right
        2: (0, -1),  # down
        3: (-1, 0)   # left
    }
    
    # Fill in the policy directions
    for y in range(height):
        for x in range(width):
            state = (x, y)
            
            # Get the action for this state
            if callable(policy):
                action = policy(state)
            else:
                action = policy.get(state, 0)
            
            # Get the direction for this action
            dx, dy = directions.get(action, (0, 0))
            
            # Set the components for the arrow
            U[height - 1 - y, x] = dx
            V[height - 1 - y, x] = dy
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the policy as arrows
    ax.quiver(np.arange(width), np.arange(height), U, V, scale=20, pivot='middle')
    
    # Add grid
    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, height, 1))
    ax.set_xlim(-.5, width - .5)
    ax.set_ylim(-.5, height - .5)
    ax.invert_yaxis()  # Invert y-axis to match the grid coordinates
    ax.grid(True)
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    return fig, ax


class PolicyEvaluator:
    """
    Evaluates a policy in a given environment.
    
    This class provides methods to evaluate a policy by running multiple episodes
    and collecting statistics on the rewards obtained.
    
    Attributes:
        env: The environment to evaluate the policy in.
        episodes (int): Number of episodes to run for evaluation.
    """
    
    def __init__(self, env, episodes=100):
        """
        Initialize the policy evaluator.
        
        Args:
            env: The environment to evaluate the policy in.
            episodes (int): Number of episodes to run for evaluation. Default is 100.
        """
        self.env = env
        self.episodes = episodes
    
    def evaluate(self, policy, render=False):
        """
        Evaluate a policy by running multiple episodes and collecting statistics.
        
        Args:
            policy: The policy to evaluate (function or object with get_action method).
            render (bool): Whether to render the environment. Default is False.
        
        Returns:
            dict: Dictionary containing evaluation metrics like mean_reward, 
                 std_reward, min_reward, and max_reward.
        """
        rewards = []
        episode_lengths = []
        
        for _ in range(self.episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Get action from policy
                if callable(policy):
                    action = policy(state)
                else:
                    action = policy.get_action(state)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Render if specified
                if render:
                    self.env.render()
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        mean_length = np.mean(episode_lengths)
        
        # Return results
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'mean_length': mean_length,
            'rewards': rewards,
            'episode_lengths': episode_lengths
        } 