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
        weight_initializer = weight_initializer_class(
            vocabulary_size=vocabulary_size,
            random_state=random_state
        )
        self.parameters = NetworkParametersCollection(
            vocabulary_size=vocabulary_size,
            hidden_layer_size=hidden_layer_size,
            weight_initializer=weight_initializer
        )

    def fit(self, x, y):
        pass

    def predict(self, x):
        return [
            np.array([
                [ 0.75,  0.39,  0.06,  0.71,  0.65,  0.38,  0.16,  0.2 ,  0.3 ,  0.03],
                [ 1.55,  0.62,  0.55,  0.88,  0.86,  0.46,  1.75,  0.46,  1.09,  0.4 ],
                [ 0.19,  2.18,  0.93,  0.21,  1.03,  1.29,  0.77,  0.29,  0.26,  0.45],
                [ 2.27,  0.14,  1.01,  0.33,  0.71,  0.92,  0.47,  0.06,  0.61,  1.17]
            ])
        ]
