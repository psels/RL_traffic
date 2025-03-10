class Learner:
    def __init__(self, state_space_size, action_space, epsilon):
        self.state_space_size = state_space_size
        self.action_space = action_space
        self.epsilon = epsilon

    def act(self, state):
        return self.action_space.sample()  # ici complètement aléatoire, juste pour faire tourner le projet

    def learn(self, state, action, reward, next_state):
        pass # pour l'instant aucun apprentissage

# import numpy as np

# class Learner:
#     def __init__(self, state_space_size, action_space, epsilon, alpha=0.1, gamma=0.9):
#         # Initialisation des paramètres
#         self.state_space_size = state_space_size
#         self.action_space = action_space
#         self.epsilon = epsilon  # Probabilité d'exploration
#         self.alpha = alpha  # Taux d'apprentissage
#         self.gamma = gamma  # Facteur de réduction (discount factor)

#         # Initialisation de la table Q (avec des valeurs initiales)
#         self.Q = np.zeros((state_space_size, action_space.n))  # Utilisation de action_space.n

#     def state_to_index(self, state):
#         return hash(tuple(state)) % self.Q.shape[0]

#     def act(self, state):
#         # Exploration vs exploitation : epsilon-greedy
#         if np.random.rand() < self.epsilon:
#             return self.action_space.sample()  # Exploration : choisir une action aléatoire
#         else:
#             state_index = self.state_to_index(state)  # Correction ici
#             return np.argmax(self.Q[state_index])

#     def learn(self, state, action, reward, next_state):
#         state_index = self.state_to_index(state)  # Correction ici
#         next_state_index = self.state_to_index(next_state)  # Correction ici

#         best_next_action = np.argmax(self.Q[next_state_index])
#         td_target = reward + self.gamma * self.Q[next_state_index, best_next_action]
#         td_error = td_target - self.Q[state_index, action]

#         self.Q[state_index, action] += self.alpha * td_error

#     def show_Q_table(self):
#         print("Q-table:")
#         for state_index in range(self.Q.shape[0]):  # Loop through all possible states
#             print(f"State {state_index}: {self.Q[state_index]}")

