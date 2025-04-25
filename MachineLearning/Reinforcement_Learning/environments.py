import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random


class Environment:
    """
    Base class for reinforcement learning environments.
    
    Provides the basic interface that all environments should implement.
    """
    
    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.
        
        Returns:
            Initial observation of the environment.
        """
        raise NotImplementedError
    
    def step(self, action):
        """
        Take an action in the environment and return the next observation, reward, done flag, and info.
        
        Args:
            action: The action to take in the environment.
            
        Returns:
            tuple: (next_state, reward, done, info)
                next_state: The next state of the environment.
                reward: The reward received for taking the action.
                done: Whether the episode has terminated.
                info: Additional information about the environment.
        """
        raise NotImplementedError
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The mode for rendering. Default is 'human'.
        """
        raise NotImplementedError
    
    def close(self):
        """
        Clean up the environment's resources.
        """
        pass


class GridWorld(Environment):
    """
    A customizable grid world environment for reinforcement learning.
    
    The grid world consists of a rectangular grid where the agent can move in
    four directions: up, right, down, and left. The grid can contain:
    - Empty cells that the agent can move to
    - Walls that block the agent's movement
    - Goal states that give positive rewards and optionally end the episode
    - Trap states that give negative rewards and optionally end the episode
    
    Attributes:
        width (int): Width of the grid.
        height (int): Height of the grid.
        start_pos (tuple): Starting position of the agent (x, y).
        goal_pos (list): List of goal positions, each as a tuple (x, y).
        trap_pos (list): List of trap positions, each as a tuple (x, y).
        wall_pos (list): List of wall positions, each as a tuple (x, y).
        current_pos (tuple): Current position of the agent (x, y).
        max_steps (int): Maximum number of steps per episode.
        step_count (int): Current step count in the episode.
        goal_reward (float): Reward for reaching a goal state.
        trap_reward (float): Reward for falling into a trap.
        step_reward (float): Reward for each step taken.
        goal_terminal (bool): Whether reaching a goal ends the episode.
        trap_terminal (bool): Whether falling into a trap ends the episode.
    """
    
    # Action space constants
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, width=5, height=5, start_pos=(0, 0), goal_pos=None, trap_pos=None, 
                 wall_pos=None, goal_reward=1.0, trap_reward=-1.0, step_reward=-0.01, 
                 goal_terminal=True, trap_terminal=True, max_steps=100):
        """
        Initialize the grid world environment.
        
        Args:
            width (int): Width of the grid. Default is 5.
            height (int): Height of the grid. Default is 5.
            start_pos (tuple): Starting position of the agent (x, y). Default is (0, 0).
            goal_pos (list): List of goal positions, each as a tuple (x, y). Default is [(width-1, height-1)].
            trap_pos (list): List of trap positions, each as a tuple (x, y). Default is [].
            wall_pos (list): List of wall positions, each as a tuple (x, y). Default is [].
            goal_reward (float): Reward for reaching a goal state. Default is 1.0.
            trap_reward (float): Reward for falling into a trap. Default is -1.0.
            step_reward (float): Reward for each step taken. Default is -0.01.
            goal_terminal (bool): Whether reaching a goal ends the episode. Default is True.
            trap_terminal (bool): Whether falling into a trap ends the episode. Default is True.
            max_steps (int): Maximum number of steps per episode. Default is 100.
        """
        self.width = width
        self.height = height
        self.start_pos = start_pos
        
        # Set default goal position if not provided
        if goal_pos is None:
            self.goal_pos = [(width - 1, height - 1)]
        else:
            self.goal_pos = goal_pos
        
        self.trap_pos = [] if trap_pos is None else trap_pos
        self.wall_pos = [] if wall_pos is None else wall_pos
        
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.step_reward = step_reward
        
        self.goal_terminal = goal_terminal
        self.trap_terminal = trap_terminal
        
        self.max_steps = max_steps
        self.current_pos = None
        self.step_count = 0
        
        # For visualization
        self.fig = None
        self.ax = None
        
        # Initialize the environment
        self.reset()
    
    def reset(self):
        """
        Reset the environment to the starting position.
        
        Returns:
            tuple: The initial state (x, y).
        """
        self.current_pos = self.start_pos
        self.step_count = 0
        return self.current_pos
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action (int): The action to take (0: up, 1: right, 2: down, 3: left).
            
        Returns:
            tuple: (next_state, reward, done, info)
                next_state (tuple): The next state (x, y).
                reward (float): The reward received.
                done (bool): Whether the episode has ended.
                info (dict): Additional information.
        """
        self.step_count += 1
        
        # Current position
        x, y = self.current_pos
        
        # Calculate next position based on action
        if action == self.UP:
            next_pos = (x, min(y + 1, self.height - 1))
        elif action == self.RIGHT:
            next_pos = (min(x + 1, self.width - 1), y)
        elif action == self.DOWN:
            next_pos = (x, max(y - 1, 0))
        elif action == self.LEFT:
            next_pos = (max(x - 1, 0), y)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Check if next position is a wall
        if next_pos in self.wall_pos:
            next_pos = self.current_pos  # Stay in the current position
        
        # Update position
        self.current_pos = next_pos
        
        # Calculate reward and check if done
        reward = self.step_reward  # Default reward for a step
        done = False
        
        # Check if reached a goal
        if self.current_pos in self.goal_pos:
            reward += self.goal_reward
            done = self.goal_terminal
        
        # Check if fallen into a trap
        elif self.current_pos in self.trap_pos:
            reward += self.trap_reward
            done = self.trap_terminal
        
        # Check if maximum steps reached
        if self.step_count >= self.max_steps:
            done = True
        
        # Additional information
        info = {
            'step_count': self.step_count
        }
        
        return self.current_pos, reward, done, info
    
    def render(self, mode='human'):
        """
        Render the grid world.
        
        Args:
            mode (str): The mode for rendering. Default is 'human'.
            
        Returns:
            object: Rendering object depending on the mode.
        """
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        self.ax.clear()
        
        # Set up the grid
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_xticks(range(self.width + 1))
        self.ax.set_yticks(range(self.height + 1))
        self.ax.grid(True)
        
        # Draw walls
        for x, y in self.wall_pos:
            self.ax.add_patch(Rectangle((x, y), 1, 1, fill=True, color='gray'))
        
        # Draw goals
        for x, y in self.goal_pos:
            self.ax.add_patch(Rectangle((x, y), 1, 1, fill=True, color='green', alpha=0.5))
        
        # Draw traps
        for x, y in self.trap_pos:
            self.ax.add_patch(Rectangle((x, y), 1, 1, fill=True, color='red', alpha=0.5))
        
        # Draw agent
        x, y = self.current_pos
        self.ax.add_patch(Rectangle((x, y), 1, 1, fill=True, color='blue', alpha=0.5))
        
        # Set title
        self.ax.set_title(f"Step: {self.step_count}")
        
        # Draw
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        if mode == 'rgb_array':
            # Convert the plot to a numpy array for use in other environments
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data
    
    def close(self):
        """
        Close the rendering window.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


class CartPole(Environment):
    """
    A simple implementation of the CartPole environment.
    
    The CartPole consists of a pole attached by an unactuated joint to a cart.
    The cart moves along a frictionless track. The goal is to balance the pole
    on the cart by applying forces to the cart.
    
    Attributes:
        gravity (float): Acceleration due to gravity.
        masscart (float): Mass of the cart.
        masspole (float): Mass of the pole.
        length (float): Half the length of the pole.
        force_mag (float): Magnitude of the force applied to the cart.
        tau (float): Time step size.
        theta_threshold (float): Angle at which the episode terminates.
        x_threshold (float): Position beyond which the episode terminates.
        max_steps (int): Maximum number of steps per episode.
        state (array): Current state of the environment [x, x_dot, theta, theta_dot].
        step_count (int): Current step count in the episode.
    """
    
    def __init__(self, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0,
                 tau=0.02, theta_threshold=12 * 2 * np.pi / 360, x_threshold=2.4, max_steps=500):
        """
        Initialize the CartPole environment.
        
        Args:
            gravity (float): Acceleration due to gravity. Default is 9.8.
            masscart (float): Mass of the cart. Default is 1.0.
            masspole (float): Mass of the pole. Default is 0.1.
            length (float): Half the length of the pole. Default is 0.5.
            force_mag (float): Magnitude of the force applied to the cart. Default is 10.0.
            tau (float): Time step size. Default is 0.02.
            theta_threshold (float): Angle at which the episode terminates. Default is 12 degrees.
            x_threshold (float): Position beyond which the episode terminates. Default is 2.4.
            max_steps (int): Maximum number of steps per episode. Default is 500.
        """
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masscart + self.masspole
        self.length = length  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag
        self.tau = tau  # seconds between state updates
        
        # Angle and position thresholds for termination
        self.theta_threshold_radians = theta_threshold
        self.x_threshold = x_threshold
        
        # Maximum steps
        self.max_steps = max_steps
        
        # State and step count
        self.state = None
        self.step_count = 0
        
        # For visualization
        self.fig = None
        self.ax = None
        
        # Initialize the environment
        self.reset()
    
    def reset(self):
        """
        Reset the environment to a random initial state.
        
        Returns:
            array: The initial state [x, x_dot, theta, theta_dot].
        """
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.step_count = 0
        return np.array(self.state)
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action (int): The action to take (0: push left, 1: push right).
            
        Returns:
            tuple: (next_state, reward, done, info)
                next_state (array): The next state [x, x_dot, theta, theta_dot].
                reward (float): The reward received.
                done (bool): Whether the episode has ended.
                info (dict): Additional information.
        """
        self.step_count += 1
        
        # Extract state variables
        x, x_dot, theta, theta_dot = self.state
        
        # Get the force based on the action
        force = self.force_mag if action == 1 else -self.force_mag
        
        # Calculate acceleration
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        # Equations of motion
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Update state using Euler's method
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = (x, x_dot, theta, theta_dot)
        
        # Check if done
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.step_count >= self.max_steps
        )
        
        # Reward is 1 for each step that the pole remains upright
        reward = 1.0 if not done else 0.0
        
        # Additional information
        info = {
            'step_count': self.step_count
        }
        
        return np.array(self.state), reward, done, info
    
    def render(self, mode='human'):
        """
        Render the cart-pole system.
        
        Args:
            mode (str): The mode for rendering. Default is 'human'.
            
        Returns:
            object: Rendering object depending on the mode.
        """
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
        
        self.ax.clear()
        
        # Extract state variables
        x, x_dot, theta, theta_dot = self.state
        
        # Set up the plot
        self.ax.set_xlim(-self.x_threshold - 1, self.x_threshold + 1)
        self.ax.set_ylim(-1, 1)
        
        # Draw the cart
        cart_width = 0.5
        cart_height = 0.25
        cart = plt.Rectangle((x - cart_width/2, -cart_height/2), cart_width, cart_height, 
                             fill=True, color='blue', alpha=0.5)
        self.ax.add_patch(cart)
        
        # Draw the pole
        pole_length = 2 * self.length
        pole_end_x = x + pole_length * np.sin(theta)
        pole_end_y = pole_length * np.cos(theta)
        self.ax.plot([x, pole_end_x], [0, pole_end_y], 'k-', lw=2)
        
        # Draw the ground
        self.ax.axhline(y=-cart_height/2, color='black', linestyle='-')
        
        # Set title
        self.ax.set_title(f"Step: {self.step_count}, State: x={x:.2f}, theta={theta:.2f}")
        
        # Remove y-axis ticks and labels
        self.ax.set_yticks([])
        
        # Draw
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        
        if mode == 'rgb_array':
            # Convert the plot to a numpy array for use in other environments
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data
    
    def close(self):
        """
        Close the rendering window.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


class MultiArmedBandit(Environment):
    """
    A multi-armed bandit environment for reinforcement learning.
    
    In this environment, the agent chooses one of several arms at each time step
    and receives a reward drawn from a probability distribution associated with that arm.
    The goal is to maximize the cumulative reward over time.
    
    Attributes:
        n_arms (int): Number of arms in the bandit.
        means (array): Mean rewards for each arm.
        stds (array): Standard deviations of rewards for each arm.
        max_steps (int): Maximum number of steps per episode.
        step_count (int): Current step count in the episode.
        optimal_arm (int): Index of the arm with the highest mean reward.
    """
    
    def __init__(self, n_arms=10, means=None, stds=None, max_steps=1000):
        """
        Initialize the multi-armed bandit environment.
        
        Args:
            n_arms (int): Number of arms in the bandit. Default is 10.
            means (array): Mean rewards for each arm. Default is random values in [0, 1].
            stds (array): Standard deviations of rewards for each arm. Default is 1.0 for all arms.
            max_steps (int): Maximum number of steps per episode. Default is 1000.
        """
        self.n_arms = n_arms
        
        # Set random means if not provided
        if means is None:
            self.means = np.random.uniform(0, 1, n_arms)
        else:
            self.means = means
        
        # Set constant standard deviations if not provided
        if stds is None:
            self.stds = np.ones(n_arms)
        else:
            self.stds = stds
        
        self.max_steps = max_steps
        self.step_count = 0
        
        # Find the optimal arm
        self.optimal_arm = np.argmax(self.means)
        
        # For tracking performance
        self.action_counts = np.zeros(n_arms)
        self.optimal_actions = 0
        
        # For visualization
        self.fig = None
        self.ax = None
        self.action_history = []
        self.reward_history = []
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            int: Initial state (0 for multi-armed bandits, as there's no state).
        """
        self.step_count = 0
        self.action_counts = np.zeros(self.n_arms)
        self.optimal_actions = 0
        self.action_history = []
        self.reward_history = []
        return 0  # In bandits, the state is irrelevant
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action (int): The arm to pull (0 to n_arms-1).
            
        Returns:
            tuple: (next_state, reward, done, info)
                next_state (int): The next state (always 0 for bandits).
                reward (float): The reward received.
                done (bool): Whether the episode has ended.
                info (dict): Additional information.
        """
        self.step_count += 1
        
        # Validate action
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Invalid action: {action}. Must be between 0 and {self.n_arms-1}.")
        
        # Track action
        self.action_counts[action] += 1
        self.action_history.append(action)
        
        # Check if optimal action was chosen
        if action == self.optimal_arm:
            self.optimal_actions += 1
        
        # Generate reward
        reward = np.random.normal(self.means[action], self.stds[action])
        self.reward_history.append(reward)
        
        # Check if done
        done = self.step_count >= self.max_steps
        
        # Additional information
        info = {
            'step_count': self.step_count,
            'action_counts': self.action_counts,
            'optimal_action_rate': self.optimal_actions / self.step_count,
            'regret': np.max(self.means) * self.step_count - np.sum(self.reward_history)
        }
        
        return 0, reward, done, info  # State is always 0 for bandits
    
    def render(self, mode='human'):
        """
        Render the multi-armed bandit environment.
        
        Args:
            mode (str): The mode for rendering. Default is 'human'.
            
        Returns:
            object: Rendering object depending on the mode.
        """
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Clear the plots
        self.ax[0].clear()
        self.ax[1].clear()
        
        # Plot the true mean rewards
        self.ax[0].bar(range(self.n_arms), self.means, alpha=0.7, label='True Mean Rewards')
        
        # Plot the action counts
        action_freqs = self.action_counts / max(1, np.sum(self.action_counts))
        self.ax[0].bar(range(self.n_arms), action_freqs, alpha=0.5, label='Action Frequencies')
        
        # Highlight the optimal arm
        self.ax[0].scatter([self.optimal_arm], [self.means[self.optimal_arm]], color='red', 
                           marker='*', s=200, label='Optimal Arm')
        
        # Set labels and legend
        self.ax[0].set_xlabel('Arms')
        self.ax[0].set_ylabel('Mean Reward / Action Frequency')
        self.ax[0].set_title(f'Arm Mean Rewards and Action Frequencies')
        self.ax[0].legend()
        
        # Plot the reward history
        if len(self.reward_history) > 0:
            self.ax[1].plot(self.reward_history, alpha=0.7, label='Rewards')
            
            # Calculate and plot the moving average of rewards
            window = min(100, len(self.reward_history))
            if window > 0:
                moving_avg = np.convolve(self.reward_history, np.ones(window) / window, mode='valid')
                self.ax[1].plot(range(window-1, len(self.reward_history)), moving_avg, 
                               label=f'Moving Average (window={window})')
            
            # Plot the optimal reward
            self.ax[1].axhline(y=np.max(self.means), color='r', linestyle='--', 
                              alpha=0.5, label='Optimal Mean Reward')
        
        # Set labels and legend
        self.ax[1].set_xlabel('Steps')
        self.ax[1].set_ylabel('Reward')
        self.ax[1].set_title(f'Reward History (Step {self.step_count})')
        self.ax[1].legend()
        
        # Draw
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        
        if mode == 'rgb_array':
            # Convert the plot to a numpy array for use in other environments
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data
    
    def close(self):
        """
        Close the rendering window.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None 