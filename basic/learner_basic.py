class Learner:
    def __init__(self, state_space_size, action_space, epsilon):
        self.state_space_size = state_space_size
        self.action_space = action_space
        self.epsilon = epsilon

    def act(self, state):
        return self.action_space.sample()  # ici complètement aléatoire, juste pour faire tourner le projet

    def learn(self, state, action, reward, next_state):
        pass # pour l'instant aucun apprentissage



