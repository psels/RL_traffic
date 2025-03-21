import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class DuelingDQN(tf.keras.Model):
    def __init__(self,n_inputs, n_outputs,name="DuelingDQN"):
        super(DuelingDQN, self).__init__(name=name)
        # Couche d'entrée commune
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.name=name
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

    def get_config(self):
        config = super(DuelingDQN, self).get_config()
        config.update({
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "name":self.name
        })
        return config


    @classmethod
    def from_config(cls, config):
        """Recreate model, ignoring extra arguments like 'trainable'"""
        allowed_keys = {"n_inputs", "n_outputs", "name"}  # ✅ Liste des clés attendues
        filtered_config = {k: v for k, v in config.items() if k in allowed_keys}  # ✅ Filtrer les paramètres
        return cls(**filtered_config)  # ✅ Appel sécurisé sans erreur
    # @classmethod
    # def from_config(cls, config):
        # return cls(**config)
