import numpy as np

from training_batch_generator import MiniBatchGenerator
from optimization_algorithm import GradientDescent
from loss_function import MeanSquaredError
from activation_function import SigmoidActivationFunction


class VanillaNeuralNetwork:

    def __init__(self, layer_sizes, training_batch_generator_class, loss_function_class,
            activation_function_class, optimization_algorithm_class, learning_rate, n_epochs,
            training_batch_size, holdout_data=None, random_state=123):
        self.layer_sizes = layer_sizes
        self.training_batch_generator_class = MiniBatchGenerator
        self.loss_function_class = MeanSquaredError
        self.activation_function_class = SigmoidActivationFunction
        self.optimization_algorithm_class = GradientDescent
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.training_batch_size = training_batch_size
        self.holdout_data = holdout_data
        self.random_number_generator = np.random.RandomState(random_state)
        self.bias_vectors = self._initialize_bias_vectors()
        self.weight_matrices = self._initialize_weight_matrices()

    def fit(self, X, y):
        for epoch in range(self.n_epochs):
            training_batch_generator = self.training_batch_generator_class(X=X, y=y, batch_size=self.training_batch_size,
                random_number_generator=self.random_number_generator)

            for training_batch in training_batch_generator:
                self.update_mini_batch(training_batch)
            if self.holdout_data:
                holdout_accuracy = self._validate_on_holdout_set()
                print('Epoch: {} | Accuracy: {}'.format(epoch, np.round(holdout_accuracy, 5)))

    def predict(self, X):
        activation_matrix = X
        for bias_vector, weight_matrix in zip(self.bias_vectors, self.weight_matrices):
            linear_combination = np.dot(activation_matrix, weight_matrix.T) + bias_vector
            activation_matrix = self.activation_function_class.activation_function(linear_combination)
        return activation_matrix

    def update_mini_batch(self, training_batch):
        self.weight_matrices, self.bias_vectors = self.optimization_algorithm_class(
            training_batch=training_batch,
            weight_matrices=self.weight_matrices,
            bias_vectors=self.bias_vectors,
            loss_function_class=self.loss_function_class,
            activation_function_class=self.activation_function_class,
            learning_rate=self.learning_rate
        ).run()

    def _validate_on_holdout_set(self):
        holdout_predictions = self.predict(self.holdout_data.X)
        return self.loss_function_class.accuracy(
            y_true=self.holdout_data.y,
            y_predicted=holdout_predictions
        )

    def _initialize_weight_matrices(self):
        return [self.random_number_generator.randn(next_layer_size, layer_size) for layer_size, next_layer_size \
            in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def _initialize_bias_vectors(self):
        return [self.random_number_generator.randn(layer_size) for layer_size in self.layer_sizes[1:]]
