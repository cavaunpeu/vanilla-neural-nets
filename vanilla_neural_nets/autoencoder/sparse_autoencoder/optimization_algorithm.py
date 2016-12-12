from collections import deque

import numpy as np

from vanilla_neural_nets.neural_network.optimization_algorithm import GradientDescent


class SparsityEnforcingGradientDescent(GradientDescent):

    def __init__(self, training_batch, network_layers, loss_function_class,
            activation_function_class, output_layer_activation_function_class,
            learning_rate, sparsity_constraint_class, rho, beta, rho_hat_clip_epsilon):
        super().__init__(
            training_batch=training_batch,
            network_layers=network_layers,
            loss_function_class=loss_function_class,
            activation_function_class=activation_function_class,
            output_layer_activation_function_class=output_layer_activation_function_class,
            learning_rate=learning_rate
        )
        self.sparsity_constraint_class = sparsity_constraint_class
        self.rho = rho
        self.beta = beta
        self.rho_hat_clip_epsilon = rho_hat_clip_epsilon

    def run(self):
        self._compute_sparsity()
        self._compute_gradients()
        self._update_parameters()
        return self.parameters

    def _compute_sparsity(self):
        self._feed_forward(self.X)
        self.parameters.reset_hidden_layer_sparsity_coefficients_to_zero()
        self.parameters.hidden_layer_sparsity_coefficients = self._hidden_layer_activations.mean(axis=0).clip(self.rho_hat_clip_epsilon, 1 - self.rho_hat_clip_epsilon)
        self.linear_combination_matrices = []
        self.activation_matrices = []

    def _compute_delta_matrices(self):
        output_layer_delta_matrix = self._compute_output_layer_delta_matrix()

        hidden_layer_delta_matrix = (
            np.dot(output_layer_delta_matrix, list(self.parameters.layers)[-1].weight_parameter.value) + \
            self.beta * self.sparsity_constraint_class.derivative_of_loss_function(rho=self.rho, vector_of_rho_hats=self.parameters.hidden_layer_sparsity_coefficients)
        ) * self.activation_function_class.derivative_of_activation_function(self.linear_combination_matrices[0])

        self.delta_matrices = [hidden_layer_delta_matrix, output_layer_delta_matrix]

    @property
    def _hidden_layer_activations(self):
        return self.activation_matrices[1]
