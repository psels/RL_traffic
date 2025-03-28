import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class DuelingDQN(tf.keras.Model):
    """
    Dueling DQN architecture separating value and advantage estimation.
    """
    def __init__(self, n_inputs, n_outputs, name="DuelingDQN"):
        super(DuelingDQN, self).__init__(name=name)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.name = name

        # Shared hidden layers
        self.shared_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(n_inputs,)),
            tf.keras.layers.Dense(128, activation="relu"),
        ])

        # Value stream V(s)
        self.value_stream = tf.keras.layers.Dense(1)

        # Advantage stream A(s, a)
        self.advantage_stream = tf.keras.layers.Dense(n_outputs)

    def call(self, inputs):
        """
        Forward pass: compute Q(s, a) using dueling streams.
        """
        x = self.shared_layers(inputs)

        V = self.value_stream(x)                   # Estimate of state value
        A = self.advantage_stream(x)               # Advantage for each action
        Q = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))  # Combine streams

        return Q

    def get_config(self):
        """
        Return model configuration for serialization.
        """
        config = super(DuelingDQN, self).get_config()
        config.update({
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "name": self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Reconstruct the model from config, safely ignoring extra arguments.
        """
        allowed_keys = {"n_inputs", "n_outputs", "name"}
        filtered_config = {k: v for k, v in config.items() if k in allowed_keys}
        return cls(**filtered_config)
