import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self, n_inputs, n_outputs):
        super(DQN, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(n_inputs,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(n_outputs)  # Chaque sortie correspond à une action Q(s, a)
        ])

    def call(self, inputs):
        return self.model(inputs)  # Retourne Q(s, a)
