from collections import deque

import numpy as np

from recurrent_neural_network.loss_function import CrossEntropyLoss
from recurrent_neural_network.parameter_object import NetworkParametersCollection


class VanillaRecurrentNeuralNetwork:

    # you should pass in a _feed_forward function from inside of your network to the optimization routine, so you don't repeat yourself.

    def __init__(self, vocabulary_size, hidden_layer_size, backprop_through_time_steps,
        optimization_algorithm_class, weight_initializer_class, learning_rate, n_epochs,
        random_state, loss_function_class=CrossEntropyLoss):
        self.backprop_through_time_steps = backprop_through_time_steps
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

    def compute_gradients(self, x, y):
        softmax_outputs, hidden_state = self._feed_forward(x)
        time_steps = np.arange(len(x))

        # compute gradients
        dJdW_hy = np.zeros_like(self.parameters.W_hy.gradient)
        dJdW_hh = np.zeros_like(self.parameters.W_hh.gradient)
        dJdW_xh = np.zeros_like(self.parameters.W_xh.gradient)

        for t in time_steps[::-1]:
            label = y[t]

            # derivative of loss function w.r.t. softmax predictions
            dJdP = softmax_outputs[t]
            dJdP[label] -= 1

            # derivative of loss function w.r.t. W_hy
            dJdW_hy += np.outer(dJdP, hidden_state[t])

            # derivative of loss function w.r.t. hidden state
            dJdH = dJdP @ self.parameters.W_hy.value

            # initialize dJdH_parent
            dJdH_parent = dJdH

            # back-propagate through time
            back_prop_through_time_steps = np.arange( max(0, t-self.backprop_through_time_steps), t+1)
            for t_ in back_prop_through_time_steps[::-1]:

                # derivative of loss function w.r.t. hidden-layer input
                dJdZ = dJdH_parent * (1 - hidden_state[t_]**2)

                # derivative of loss function w.r.t. W_hh
                dJdW_hh += np.outer(dJdZ, hidden_state[t_-1])

                # derivative of loss function w.r.t. W_xh
                input_vector = np.arange(self.vocabulary_size) == x[t_]
                dJdW_xh += np.outer(dJdZ, input_vector)

                # derivative of loss function w.r.t. *previous* hidden state
                dJdH_parent = dJdZ @ self.parameters.W_hh.value

        # update gradients
        self.parameters.W_hy.gradient = dJdW_hy
        self.parameters.W_hh.gradient = dJdW_hh
        self.parameters.W_xh.gradient = dJdW_xh

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
        hidden_state.rotate(-1)

        return np.array(softmax_outputs), np.array(hidden_state)

    def _compute_softmax(self, vector):
        exponentiated_terms = np.exp(vector)
        return exponentiated_terms / exponentiated_terms.sum()
