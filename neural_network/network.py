import numpy as np

from neural_network.training_batch_generator import MiniBatchGenerator
from neural_network.optimization_algorithm import GradientDescent
from neural_network.loss_function import MeanSquaredError
from neural_network.activation_function import SigmoidActivationFunction
from neural_network.layer_object import NetworkLayersCollection


class VanillaNeuralNetwork:

    def __init__(self, layer_sizes, training_batch_generator_class, loss_function_class,
            activation_function_class, optimization_algorithm_class, learning_rate, n_epochs,
            training_batch_size, output_layer_activation_function_class=None,
            holdout_data=None, random_state=123):
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
        self.network_layers = NetworkLayersCollection(layer_sizes=layer_sizes,
            random_number_generator=self.random_number_generator)

    def fit(self, X, y):
        for epoch in range(self.n_epochs):
            training_batch_generator = self.training_batch_generator_class(X=X, y=y, batch_size=self.training_batch_size,
                random_number_generator=self.random_number_generator)

            for training_batch in training_batch_generator:
                self._update_network_layers_with_training_batch(training_batch)
            if self.holdout_data:
                holdout_accuracy = self._validate_on_holdout_set()
                print('Epoch: {} | Accuracy: {}'.format(epoch, np.round(holdout_accuracy, 5)))

    def predict(self, X):
        activation_matrix = X
        for layer in self.network_layers:
            activation_function_class = self.output_layer_activation_function_class if layer.output_layer\
                else self.activation_function_class

            linear_combination = np.dot(activation_matrix, layer.weight_matrix.T) + layer.bias_vector
            activation_matrix = activation_function_class.activation_function(linear_combination)
        return activation_matrix

    def _update_network_layers_with_training_batch(self, training_batch):
        self.network_layers = self.optimization_algorithm_class(
            training_batch=training_batch,
            network_layers=self.network_layers,
            loss_function_class=self.loss_function_class,
            activation_function_class=self.activation_function_class,
            output_layer_activation_function_class=self.output_layer_activation_function_class,
            learning_rate=self.learning_rate
        ).run()

    def _validate_on_holdout_set(self):
        holdout_predictions = self.predict(self.holdout_data.X)
        return self.loss_function_class.accuracy(
            y_true=self.holdout_data.y,
            y_predicted=holdout_predictions
        )
