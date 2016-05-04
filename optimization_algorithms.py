from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np


class BaseOptimizationAlgorithm(metaclass=ABCMeta):
    
    @abstractmethod
    def run(self):
        pass


class GradientDescent(BaseOptimizationAlgorithm):
    
    def __init__(self, weight_matrices, bias_vectors, linear_combinations, activations, y, 
                 activation_function_class, loss_function_class, learning_rate):
        self.weight_matrices = weight_matrices
        self.bias_vectors = bias_vectors
        self.linear_combinations = linear_combinations
        self.activations = activations
        self.activation_function_class = activation_function_class
        self.loss_function = loss_function_class(y_true=y, y_predicted=self.activations[-1])
        self.learning_rate = learning_rate
        self.batch_size = len(y) 
        
    def run(self):
        delta_matrices = self._compute_delta_matrices()
        updated_weight_matrices = self._update_weight_matrices(delta_matrices)
        updated_bias_vectors = self._update_bias_vectors(delta_matrices)
        return updated_weight_matrices, updated_bias_vectors
    
    def _compute_delta_matrices(self):
        output_layer_delta_matrix = self._compute_output_layer_delta_matrix()
        delta_matrices = deque([output_layer_delta_matrix])
        for linear_combination, weight_matrix in zip(reversed(self.linear_combinations[:-1]), reversed(self.weight_matrices)):
            delta_matrix = np.dot(delta_matrices[0], weight_matrix.T) * \
                self.activation_function_class.derivative_of_activation_function(linear_combination)
            delta_matrices.appendleft(delta_matrix)
        return delta_matrices
    
    def _compute_output_layer_delta_matrix(self):
        return self.loss_function.derivative_of_loss_function * \
            self.activation_function_class.derivative_of_activation_function(self.linear_combinations[-1])
        
    def _update_weight_matrices(self, delta_matrices):
        weight_gradient_matrices = self._compute_weight_gradient_matrices(delta_matrices)
        return [weight_matrix + (-self.learning_rate*weight_gradient_matrix/self.batch_size) for weight_matrix, \
            weight_gradient_matrix in zip(self.weight_matrices, weight_gradient_matrices)]
    
    def _compute_weight_gradient_matrices(self, delta_matrices):
        weight_gradient_matrices = deque()
        for activation_matrix, delta_matrix in zip(reversed(self.activations[:-1]), reversed(delta_matrices)):
            weight_gradient_matrices.appendleft(np.dot(activation_matrix.T, delta_matrix))
        return weight_gradient_matrices
    
    def _update_bias_vectors(self, delta_matrices):
        bias_gradient_vectors = self._compute_bias_gradient_vectors(delta_matrices)
        return [bias_vector + (-self.learning_rate*bias_gradient_vector/self.batch_size) for bias_vector, \
            bias_gradient_vector in zip(self.bias_vectors, bias_gradient_vectors)]
    
    def _compute_bias_gradient_vectors(self, delta_matrices):        
        return [delta_matrix.sum(axis=0) for delta_matrix in delta_matrices] 
