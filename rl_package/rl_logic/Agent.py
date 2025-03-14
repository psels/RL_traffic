from rl_package.params import *
from collections import deque
import numpy as np
import tensorflow as tf
from rl_package.rl_algorithms.model_DQN import DQN
from rl_package.rl_algorithms.model_DuelingDQN import DuelingDQN

class AgentSumo():

    def __init__(self,type_model,n_inputs, n_outputs,mem_size=MEMORY_MAX_SIZE, epsilon=1.0, epsilon_min=0.01,
                  discount_factor=0.5):
        self.type_model=type_model
        self.model_action=None
        self.model_target=None
        self.n_inputs=n_inputs
        self.n_outputs =n_outputs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.replay_buffer = deque(maxlen=mem_size)
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def build_model(self):
        if self.type_model == "DQN":
            self.model_action = DQN(self.n_inputs,self.n_outputs)
        elif self.type_model =='2DQN':
            self.model_action = DQN(self.n_inputs,self.n_outputs)
            self.model_target = DQN(self.n_inputs,self.n_outputs)
            self.model_target.set_weights(self.model_action.get_weights())
        elif self.type_model == '3DQN':
            self.model_action = DuelingDQN(self.n_inputs,self.n_outputs)
            self.model_target = DuelingDQN(self.n_inputs,self.n_outputs)
            self.model_target.set_weights(self.model_action.get_weights())


    def epsilon_greedy_policy(self,state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)  # random action
        else:
            Q_values = self.model_action.predict(state[np.newaxis], verbose=0)[0]
            return Q_values.argmax()  # optimal action according to the DQN

    def add_to_memory(self, state,action,reward, next_state): ######RAJOUTER LE DONE
        '''Adds a play to the experience replay memory buffer'''
        # self.memory = sorted(self.memory, key=lambda x: x[2])
        # if len(self.memory) == MEMORY_MAX_SIZE:
        #     self.memory.pop(0)
        self.replay_buffer.append((state, action, reward, next_state))

    def training_step(self,batch_size=32):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states = experiences  # a changer

        if self.type_model=='DQN':
            next_Q_values = self.model_action.predict(next_states, verbose=0)
        else :
            next_Q_values = self.model_action.predict(next_states, verbose=0)  # ≠ target.predict()
            best_next_actions = next_Q_values.argmax(axis=1)
            next_mask = tf.one_hot(best_next_actions, self.n_outputs).numpy()
            max_next_Q_values = (self.target.predict(next_states, verbose=0) * next_mask
                                ).sum(axis=1)
            next_Q_values = self.target.predict(next_states, verbose=0)

        max_next_Q_values = next_Q_values.max(axis=1)
        target_Q_values = rewards + self.discount_factor * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model_action(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model_action.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model_action.trainable_variables))


    def sample_experiences(self,batch_size):
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(4)
        ]
        return states, actions, rewards, next_states
