import numpy as np

from vanilla_neural_nets.neural_network.layer_object import NetworkLayersCollection


class NetworkLayersCollectionWithSparsity(NetworkLayersCollection):

    def __init__(self, layer_sizes, weight_initializer, bias_initializer):
        self.layer_sizes = layer_sizes
        self.reset_hidden_layer_sparsity_coefficients_to_zero()
        super().__init__(layer_sizes=layer_sizes, weight_initializer=weight_initializer, bias_initializer=bias_initializer)

    def reset_hidden_layer_sparsity_coefficients_to_zero(self):
        self.hidden_layer_sparsity_coefficients = np.zeros(self.layer_sizes[1])
