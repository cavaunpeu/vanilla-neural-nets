import sys

import numpy as np

from vanilla_neural_nets.neural_network.network import VanillaNeuralNetwork
from vanilla_neural_nets.autoencoder.sparse_autoencoder.layer_object import NetworkLayersCollectionWithSparsity


class VanillaSparseAutoencoder(VanillaNeuralNetwork):

    def __init__(self, layer_sizes, training_batch_generator_class, loss_function_class,
            activation_function_class, optimization_algorithm_class, learning_rate, n_epochs,
            training_batch_size, weight_initializer, bias_initializer, sparsity_constraint_class,
            rho, beta, output_layer_activation_function_class=None, holdout_data=None, random_state=123,
            rho_hat_clip_epsilon=1e-15):
        if len(layer_sizes) > 3:
            sys.exit('Autoencoder should have only one hidden layer')

        self.training_batch_generator_class = training_batch_generator_class
        self.loss_function_class = loss_function_class
        self.activation_function_class = activation_function_class
        self.output_layer_activation_function_class = output_layer_activation_function_class or activation_function_class
        self.optimization_algorithm_class = optimization_algorithm_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.training_batch_size = training_batch_size
        self.holdout_data = holdout_data
        self.random_number_generator = np.random.RandomState(random_state)
        self.parameters = NetworkLayersCollectionWithSparsity(
            layer_sizes=layer_sizes,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer
        )
        self.sparsity_constraint_class = sparsity_constraint_class
        self.rho = rho
        self.beta = beta
        self.rho_hat_clip_epsilon = rho_hat_clip_epsilon

    def predict(self, x):
        activation_matrices = [x]
        for layer in self.parameters.layers:
            activation_function_class = self.output_layer_activation_function_class if layer.is_output_layer\
                else self.activation_function_class

            linear_combination = np.dot(activation_matrices[-1], layer.weight_parameter.value.T) + layer.bias_parameter.value
            activation_matrices.append(activation_function_class.activation_function(linear_combination))

        hidden_layer_activations = activation_matrices[1]
        return activation_matrices[-1], hidden_layer_activations.mean(axis=0).clip(self.rho_hat_clip_epsilon, 1 - self.rho_hat_clip_epsilon)

    def generate(self, hidden_layer_activations):
        linear_combination = np.dot(hidden_layer_activations, self.parameters.weight_parameters[-1].value.T) + self.parameters.bias_parameters[-1].value
        return self.output_layer_activation_function_class.activation_function(linear_combination)

    def _update_network_layers_with_training_batch(self, training_batch):
        return self.optimization_algorithm_class(
            training_batch=training_batch,
            network_layers=self.parameters,
            loss_function_class=self.loss_function_class,
            activation_function_class=self.activation_function_class,
            output_layer_activation_function_class=self.output_layer_activation_function_class,
            learning_rate=self.learning_rate,
            sparsity_constraint_class=self.sparsity_constraint_class,
            rho=self.rho,
            beta=self.beta,
            rho_hat_clip_epsilon=self.rho_hat_clip_epsilon
        ).run()
