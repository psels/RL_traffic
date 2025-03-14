import tensorflow as tf

class DuelingDQN(tf.keras.Model):
    def __init__(self,n_inputs, n_outputs):
        super(DuelingDQN, self).__init__()
        # Couche d'entrée commune
        self.shared_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(n_inputs,)),
            tf.keras.layers.Dense(128, activation="relu"),
        ])

        # **Stream Valeur** V(s)
        self.value_stream = tf.keras.layers.Dense(1)  # Une seule sortie : la valeur de l'état

        # **Stream Avantage** A(s, a)
        self.advantage_stream = tf.keras.layers.Dense(n_outputs)# Une sortie par action

    def call(self, inputs):
        x = self.shared_layers(inputs)

        V = self.value_stream(x)  # Calcul de la valeur de l’état
        A = self.advantage_stream(x)  # Calcul des avantages

        # Normalisation de A pour éviter le biais
        Q = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))

        return Q
