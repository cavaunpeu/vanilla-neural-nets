from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np


class BaseOptimizationAlgorithm(metaclass=ABCMeta):

    @abstractmethod
    def run(self):
        pass


class GradientDescent:

    def __init__(self, training_batch, network_layers, loss_function_class,
            activation_function_class, output_layer_activation_function_class, learning_rate):
        self.X = training_batch.X
        self.y = training_batch.y
        self.batch_size = len(self.X)
        self.network_layers = network_layers
        self.loss_function_class = loss_function_class
        self.activation_function_class = activation_function_class
        self.output_layer_activation_function_class = output_layer_activation_function_class
        self.learning_rate = learning_rate
        self.linear_combination_matrices = []
        self.activation_matrices = []
        self.delta_matrices = []

    def run(self):
        self._feed_forward(self.X)
        self._compute_delta_matrices()
        self.network_layers.weight_matrices = self._compute_updated_weight_matrices()
        self.network_layers.bias_vectors = self._compute_updated_bias_vectors()
        return self.network_layers

    def _feed_forward(self, X):
        self.activation_matrices.append(X)
        for layer in self.network_layers:
            activation_function_class = self.output_layer_activation_function_class if layer.output_layer\
                else self.activation_function_class

            linear_combination = np.dot(self.activation_matrices[-1], layer.weight_matrix.T) + layer.bias_vector
            self.linear_combination_matrices.append(linear_combination)
            activation_matrix = activation_function_class.activation_function(linear_combination)
            self.activation_matrices.append(activation_matrix)

    def _compute_delta_matrices(self):
        output_layer_delta_matrix = self._compute_output_layer_delta_matrix()
        delta_matrices = deque([output_layer_delta_matrix])
        for linear_combination, layer in zip(reversed(self.linear_combination_matrices[:-1]), reversed(list(self.network_layers))):
            delta_matrix = np.dot(delta_matrices[0], layer.weight_matrix) * \
                self.activation_function_class.derivative_of_activation_function(linear_combination)
            delta_matrices.appendleft(delta_matrix)
        self.delta_matrices = delta_matrices

    def _compute_output_layer_delta_matrix(self):
        return self.loss_function_class.derivative_of_loss_function(y_true=self.y, y_predicted=self.activation_matrices[-1]) * \
            self.output_layer_activation_function_class.derivative_of_activation_function(self.linear_combination_matrices[-1])

    def _compute_weight_gradient_matrices(self):
        weight_gradient_matrices = deque()
        for activation_matrix, delta_matrix in zip(reversed(self.activation_matrices[:-1]), reversed(self.delta_matrices)):
            weight_gradient_matrix = np.dot(delta_matrix.T, activation_matrix)
            weight_gradient_matrices.appendleft(weight_gradient_matrix)
        return weight_gradient_matrices

    def _compute_bias_gradient_vectors(self):
        return [delta_matrix.sum(axis=0) for delta_matrix in self.delta_matrices]

    def _compute_updated_weight_matrices(self):
        weight_gradient_matrices = self._compute_weight_gradient_matrices()
        return [layer.weight_matrix + (-self.learning_rate*weight_gradient_matrix/self.batch_size) for layer, \
            weight_gradient_matrix in zip(self.network_layers, weight_gradient_matrices)]

    def _compute_updated_bias_vectors(self):
        bias_gradient_vectors = self._compute_bias_gradient_vectors()
        return [layer.bias_vector + (-self.learning_rate*bias_gradient_vector/self.batch_size) for layer, \
            bias_gradient_vector in zip(self.network_layers, bias_gradient_vectors)]
