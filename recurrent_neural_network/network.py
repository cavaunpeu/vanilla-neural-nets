from collections import deque

import numpy as np

from recurrent_neural_network.loss_function import CrossEntropyLoss
from recurrent_neural_network.parameter_object import NetworkParametersCollection


class VanillaRecurrentNeuralNetwork:

    def __init__(self, vocabulary_size, hidden_layer_size, backprop_through_time_steps,
        optimization_algorithm_class, weight_initializer_class, learning_rate, n_epochs,
        random_state, loss_function_class=CrossEntropyLoss):
        self.backprop_through_time_steps=backprop_through_time_steps
        self.optimization_algorithm_class = optimization_algorithm_class
        self.loss_function_class = loss_function_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.hidden_layer_size = hidden_layer_size
        self.vocabulary_size = vocabulary_size
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
                    learning_rate=self.learning_rate,
                    backprop_through_time_steps=self.backprop_through_time_steps,
                    vocabulary_size=self.vocabulary_size,
                    parameters=self.parameters
                ).run()

    def compute_gradients(self, x, y):
        return self.optimization_algorithm_class(
                    x=x,
                    y=y,
                    feed_forward_method=self._feed_forward,
                    learning_rate=self.learning_rate,
                    backprop_through_time_steps=self.backprop_through_time_steps,
                    vocabulary_size=self.vocabulary_size,
                    parameters=self.parameters
            ).compute_gradients()

    def predict(self, x):
        softmax_outputs, hidden_state = self._feed_forward(x)
        return softmax_outputs

    def _feed_forward(self, x):
        time_steps = len(x)
        initial_hidden_state = np.zeros(self.hidden_layer_size)
        hidden_state = deque([initial_hidden_state])
        softmax_outputs = deque()

        for t in np.arange(time_steps):
            hidden_state.append(
                np.tanh( self.parameters.W_xh.value[:, x[t]] + self.parameters.W_hh.value @ hidden_state[-1] )
            )
            softmax_outputs.append(
                self._compute_softmax( self.parameters.W_hy.value @ hidden_state[-1] )
            )
        # move initial hidden state to end of deque, such that it is later our
        # `hidden_state[t-1]` at t=0
        hidden_state.rotate(-1)

        return np.array(softmax_outputs), np.array(hidden_state)

    def _compute_softmax(self, vector):
        exponentiated_terms = np.exp(vector)
        return exponentiated_terms / exponentiated_terms.sum()
