from collections import deque

import numpy as np


class GradientDescent:

    def __init__(self, training_batch, network_layers, loss_function_class,
            activation_function_class, output_layer_activation_function_class, learning_rate):
        self.X = training_batch.X
        self.y = training_batch.y
        self.batch_size = len(self.X)
        self.parameters = network_layers
        self.loss_function_class = loss_function_class
        self.activation_function_class = activation_function_class
        self.output_layer_activation_function_class = output_layer_activation_function_class
        self.learning_rate = learning_rate
        self.linear_combination_matrices = []
        self.activation_matrices = []
        self.delta_matrices = []

    def run(self):
        self._compute_gradients()
        self._update_parameters()
        return self.parameters

    def _compute_gradients(self):
        self._feed_forward(self.X)
        self.parameters.reset_gradients_to_zero()
        self._compute_delta_matrices()
        self._compute_weight_parameter_gradients()
        self._compute_bias_parameter_gradients()

    def _update_parameters(self):
        self.parameters = self._compute_updated_weight_and_bias_parameters()

    def _feed_forward(self, X):
        self.activation_matrices.append(X)
        for layer in self.parameters.layers:
            activation_function_class = self.output_layer_activation_function_class if layer.is_output_layer\
                else self.activation_function_class

            linear_combination = np.dot(self.activation_matrices[-1], layer.weight_parameter.value.T) + layer.bias_parameter.value
            self.linear_combination_matrices.append(linear_combination)
            activation_matrix = activation_function_class.activation_function(linear_combination)
            self.activation_matrices.append(activation_matrix)

    def _compute_delta_matrices(self):
        output_layer_delta_matrix = self._compute_output_layer_delta_matrix()
        delta_matrices = deque([output_layer_delta_matrix])
        for linear_combination, layer in zip(reversed(self.linear_combination_matrices[:-1]), reversed(list(self.parameters.layers))):
            delta_matrix = np.dot(delta_matrices[0], layer.weight_parameter.value) * \
                self.activation_function_class.derivative_of_activation_function(linear_combination)
            delta_matrices.appendleft(delta_matrix)
        self.delta_matrices = delta_matrices

    def _compute_output_layer_delta_matrix(self):
        return self.loss_function_class.derivative_of_loss_function(y_true=self.y, y_predicted=self.activation_matrices[-1]) * \
            self.output_layer_activation_function_class.derivative_of_activation_function(self.linear_combination_matrices[-1])

    def _compute_weight_parameter_gradients(self):
        for layer, (activation_matrix, delta_matrix) in zip(self.parameters.layers, zip(self.activation_matrices[:-1], self.delta_matrices)):
            layer.weight_parameter.gradient += np.dot(delta_matrix.T, activation_matrix)

    def _compute_bias_parameter_gradients(self):
        for layer, delta_matrix in zip(self.parameters.layers, self.delta_matrices):
            layer.bias_parameter.gradient += delta_matrix.sum(axis=0)

    def _compute_updated_weight_and_bias_parameters(self):
        for layer in self.parameters.layers:
            layer.weight_parameter.value -= self.learning_rate * layer.weight_parameter.gradient / self.batch_size
            layer.bias_parameter.value -= self.learning_rate * layer.bias_parameter.gradient / self.batch_size
        return self.parameters
