import numpy as np


class _NetworkWeightParameter:

    def __init__(self, name, first_dimension, second_dimension, weight_initializer):
        self.name = name
        self.value = weight_initializer.initialize(
            first_dimension=first_dimension,
            second_dimension=second_dimension
        )
        self.reset_gradient_to_zero()

    def reset_gradient_to_zero(self):
        self.gradient = np.zeros_like(self.value)


class _NetworkBiasParameter:

    def __init__(self, name, size):
        self.name = name
        self.value = np.zeros(size)
        self.reset_gradient_to_zero()

    def reset_gradient_to_zero(self):
        self.gradient = np.zeros_like(self.value)
