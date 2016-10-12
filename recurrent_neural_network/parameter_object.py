import numpy as np


class NetworkWeightParameter:

    def __init__(self, name, first_dimension, second_dimension, weight_initializer):
        self.name = name
        self.value = weight_initializer.initialize(
            first_dimension=first_dimension,
            second_dimension=second_dimension
        )
        self.reset_gradient_to_zero()

    def reset_gradient_to_zero(self):
        self.gradient = np.zeros_like(self.value)


class NetworkParametersCollection:

    def __init__(self, vocabulary_size, hidden_layer_size, weight_initializer):
        self.W_xh = NetworkWeightParameter(
            name='W_xh',
            first_dimension=hidden_layer_size,
            second_dimension=vocabulary_size,
            weight_initializer=weight_initializer
        )
        self.W_hh = NetworkWeightParameter(
            name='W_hh',
            first_dimension=hidden_layer_size,
            second_dimension=hidden_layer_size,
            weight_initializer=weight_initializer
        )
        self.W_hy = NetworkWeightParameter(
            name='W_hy',
            first_dimension=vocabulary_size,
            second_dimension=hidden_layer_size,
            weight_initializer=weight_initializer
        )

    def reset_gradients_to_zero(self):
        self.W_xh.reset_gradient_to_zero()
        self.W_hh.reset_gradient_to_zero()
        self.W_hy.reset_gradient_to_zero()
