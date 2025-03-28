from rl_package.params import *
from collections import deque
import numpy as np
import tensorflow as tf
from rl_package.rl_algorithms.model_DQN import DQN
from rl_package.rl_algorithms.model_DuelingDQN import DuelingDQN

class AgentSumo:
    """
    Reinforcement learning agent for traffic light control using DQN, Double DQN or Dueling DQN.
    """

    def __init__(self, type_model, n_inputs, n_outputs,
                 mem_size=MEMORY_MAX_SIZE, epsilon=1.0, epsilon_min=0.01, discount_factor=0.1):
        """
        Initializes the agent with its hyperparameters and replay buffer.

        Args:
            type_model (str): "DQN", "2DQN", or "3DQN"
            n_inputs (int): input size
            n_outputs (int): number of possible actions
            mem_size (int): replay buffer size
            epsilon (float): initial exploration rate
            discount_factor (float): gamma discount factor
        """
        self.type_model = type_model
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.model_action = None  # Main network for Q-value prediction
        self.model_target = None  # Target network (for Double/Dueling DQN)

        self.replay_buffer = deque(maxlen=mem_size)
        self.discount_factor = discount_factor

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
        self.loss_fn = tf.keras.losses.MeanSquaredError()


    def build_model(self):
        """
        Builds the action model (and target model if needed).
        Initializes weights to avoid errors with set_weights().
        """
        if self.type_model in ["DQN", "2DQN"]:
            self.model_action = DQN(self.n_inputs, self.n_outputs)
        elif self.type_model == "3DQN":
            self.model_action = DuelingDQN(self.n_inputs, self.n_outputs)

        dummy_input = tf.zeros((1, self.n_inputs))
        self.model_action(dummy_input)

        if self.type_model in ["2DQN", "3DQN"]:
            self.model_target = tf.keras.models.clone_model(self.model_action)
            self.model_target(dummy_input)
            self.model_target.set_weights(self.model_action.get_weights())


    def epsilon_greedy_policy(self, state, epsilon=0):
        """
        Chooses an action using epsilon-greedy policy.

        Args:
            state (np.array): current state
            epsilon (float): exploration rate

        Returns:
            int: chosen action
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            Q_values = self.model_action.predict(state[np.newaxis], verbose=0)[0]
            return Q_values.argmax()


    def add_to_memory(self, state, action, reward, next_state):
        """
        Adds a transition to the replay memory.

        Args:
            state (np.array)
            action (int)
            reward (float)
            next_state (np.array)
        """
        self.replay_buffer.append((state, action, reward, next_state))


    def training_step(self, batch_size=32):
        """
        Trains the action model using a batch of experiences.
        Uses target model if applicable.

        Args:
            batch_size (int): size of experience batch
        """
        states, actions, rewards, next_states = self.sample_experiences(batch_size)

        # Standard DQN
        if self.type_model == 'DQN':
            next_Q_values = self.model_action.predict(next_states, verbose=0)
            max_next_Q_values = next_Q_values.max(axis=1)
        else:
            # Double/Dueling DQN
            best_next_actions = self.model_action.predict(next_states, verbose=0).argmax(axis=1)
            next_Q_values_target = self.model_target.predict(next_states, verbose=0)
            max_next_Q_values = next_Q_values_target[np.arange(batch_size), best_next_actions]

        target_Q_values = rewards + self.discount_factor * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)

        mask = tf.one_hot(actions, self.n_outputs)

        with tf.GradientTape() as tape:
            all_Q_values = self.model_action(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model_action.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model_action.trainable_variables))


    def sample_experiences(self, batch_size):
        """
        Samples a random batch of experiences from replay buffer.

        Args:
            batch_size (int)

        Returns:
            tuple: (states, actions, rewards, next_states)
        """
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[i] for i in indices]

        states, actions, rewards, next_states = [
            np.array([experience[i] for experience in batch]) for i in range(4)
        ]
        return states, actions, rewards, next_states
