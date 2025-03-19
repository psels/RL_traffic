import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self, input_shape, n_outputs):
        super(DQN, self).__init__()

        # Réseau de neurones classique
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=input_shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(n_outputs)  # Sortie unique avec Q(s,a) pour chaque action
        ])

    def call(self, inputs):
        return self.model(inputs)  # Pas de séparation entre valeur et avantage
