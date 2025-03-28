import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class DQN(tf.keras.Model):
    """
    Standard Deep Q-Network (DQN) model using two hidden layers.
    """
    def __init__(self, n_inputs, n_outputs, name="DQN"):
        super(DQN, self).__init__(name=name)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.name = name

        # Hidden layers
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
        ])

        # Output layer: Q-values for each action
        self.output_layer = tf.keras.layers.Dense(n_outputs)

    def call(self, inputs):
        """
        Forward pass to compute Q-values from state inputs.
        """
        x = self.hidden_layers(inputs)
        Q_values = self.output_layer(x)
        return Q_values

    def get_config(self):
        """
        Return model configuration for serialization.
        """
        config = super(DQN, self).get_config()
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
