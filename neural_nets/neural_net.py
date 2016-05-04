import numpy as np


class VanillaNeuralNet:
    
    def __init__(self, layer_sizes, training_batch_generator_class, loss_function_class, activation_function_class, 
                 optimization_algorithm_class, learning_rate, n_epochs, n_batches_per_epoch, holdout_data):
        self.weight_matrices = self._initialize_weight_matrices(layer_sizes)
        self.bias_vectors = self._initialize_bias_vectors(layer_sizes)
        self.training_batch_generator_class = training_batch_generator_class
        self.loss_function_class = loss_function_class
        self.activation_function_class = activation_function_class
        self.optimization_algorithm_class = optimization_algorithm_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_batches_per_epoch = n_batches_per_epoch
        self.holdout_data = holdout_data
        
    def fit(self, X, y):
        for epoch in range(self.n_epochs):
            print('Epoch {}'.format(epoch))
            training_batch_generator = self.training_batch_generator_class(X=X, y=y, n_batches=self.n_batches_per_epoch)
            
            for training_batch in training_batch_generator:
                self._update_weights_and_biases(training_batch)
            self._validate_on_holdout_set()

    def predict(self, X):
        linear_combination_matrices, activations = self._feed_forward(X)
        return activations[-1]
    
    def _update_weights_and_biases(self, training_batch):
        linear_combinations, activations = self._feed_forward(training_batch.X)
        self._back_propagate(linear_combinations=linear_combinations, activations=activations, y=training_batch.y)

    def _feed_forward(self, X):
        activation_matrices = [X]
        linear_combination_matrices = []
        for weight_matrix, bias_vector in zip(self.weight_matrices, self.bias_vectors):
            linear_combination = np.dot(activation_matrices[-1], weight_matrix) + bias_vector
            linear_combination_matrices.append(linear_combination)
            activation_matrix = self.activation_function_class.activation_function(linear_combination)
            activation_matrices.append(activation_matrix)
        return linear_combination_matrices, activation_matrices
        
    def _back_propagate(self, linear_combinations, activations, y):
        self.weight_matrices, self.bias_vectors = self.optimization_algorithm_class(
            weight_matrices=self.weight_matrices,
            bias_vectors=self.bias_vectors,
            linear_combinations=linear_combinations,
            activations=activations,
            y=y,
            activation_function_class=self.activation_function_class,
            loss_function_class=self.loss_function_class,
            learning_rate=self.learning_rate
        ).run()
        
    def _validate_on_holdout_set(self):
        holdout_predictions = self.predict(self.holdout_data.X)
        loss_function = self.loss_function_class(
            y_true=self.holdout_data.y,
            y_predicted=holdout_predictions
        )
        print('Holdout cost: {}'.format(np.round(loss_function.cost, 5)))
        print('Holdout accuracy: {}'.format(np.round(loss_function.accuracy, 5)))
        
    @staticmethod
    def _initialize_weight_matrices(layer_sizes):
        return [np.random.randn(layer_size, next_layer_size) for layer_size, next_layer_size \
                in zip(layer_sizes[:-1], layer_sizes[1:])]
    
    @staticmethod
    def _initialize_bias_vectors(layer_sizes):
        return [np.random.randn(layer_size) for layer_size in layer_sizes[1:]]
