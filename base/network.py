from abc import ABCMeta, abstractmethod

import numpy as np

from vanilla_neural_nets.recurrent_neural_network.loss_function import CrossEntropyLoss
from vanilla_neural_nets.recurrent_neural_network.parameter_object import NetworkParametersCollection


class BaseRecurrentNeuralNetwork(metaclass=ABCMeta):

    def __init__(self, vocabulary_size, hidden_layer_size, backprop_through_time_class,
        backprop_through_time_steps, optimization_algorithm_class, weight_initializer_class,
        learning_rate, n_epochs, random_state, loss_function_class=CrossEntropyLoss, log_training_loss=False):
        self.backprop_through_time_class = backprop_through_time_class
        self.backprop_through_time_steps = backprop_through_time_steps
        self.optimization_algorithm_class = optimization_algorithm_class
        self.loss_function_class = loss_function_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.hidden_layer_size = hidden_layer_size
        self.vocabulary_size = vocabulary_size
        self.log_training_loss = log_training_loss
        weight_initializer = weight_initializer_class(
            vocabulary_size=vocabulary_size,
            random_state=random_state
        )
        self.parameters = NetworkParametersCollection(
            vocabulary_size=vocabulary_size,
            hidden_layer_size=hidden_layer_size,
            weight_initializer=weight_initializer
        )

    def fit(self, X, y):
        for epoch in range(self.n_epochs):
            for sentence, labels in zip(X, y):
                self.parameters = self.optimization_algorithm_class(
                    x=sentence,
                    y=labels,
                    feed_forward_method=self._feed_forward,
                    backprop_through_time_class=self.backprop_through_time_class,
                    learning_rate=self.learning_rate,
                    backprop_through_time_steps=self.backprop_through_time_steps,
                    vocabulary_size=self.vocabulary_size,
                    parameters=self.parameters
                ).run()
            if self.log_training_loss:
                training_loss = self._compute_training_loss(x=sentence, y_true=labels)
                print('Epoch: {} | Loss: {}'.format(epoch, np.round(training_loss, 5)))

    def predict(self, x):
        softmax_outputs, hidden_state = self._feed_forward(x)
        return softmax_outputs

    @abstractmethod
    def _feed_forward(self, x):
        pass

    def _compute_softmax(self, vector):
        exponentiated_terms = np.exp(vector)
        return exponentiated_terms / exponentiated_terms.sum()

    def _compute_training_loss(self, x, y_true):
        softmax_outputs, hidden_state = self._feed_forward(x)
        return self.loss_function_class.total_loss(y_true=y_true, y_predicted=softmax_outputs)
