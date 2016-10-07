import unittest

from recurrent_neural_network.network import VanillaRecurrentNeuralNetwork
from recurrent_neural_network.optimization_algorithm import RNNGradientDescent
from recurrent_neural_network.parameter_initialization import OneOverRootNWeightInitializer
from tests.helpers.gradient_check import RNNGradientChecker


class TestVanillaRecurrentNeuralNetwork(unittest.TestCase):

    X_TRAIN = [0, 1, 2, 3]
    Y_TRAIN = [1, 2, 3, 4]
    VOCABULARY_SIZE = 10
    
    HIDDEN_LAYER_SIZE = 3
    BACKPROP_THROUGH_TIME_STEPS = SOME_LARGE_NUMBER = 1000
    LEARNING_RATE = .005
    N_EPOCHS = 100
    RANDOM_STATE = 12345

    def test_network_passes_gradient_check(self):
        network = VanillaRecurrentNeuralNetwork(
            vocabulary_size=self.VOCABULARY_SIZE,
            hidden_layer_size=self.HIDDEN_LAYER_SIZE,
            backprop_through_time_steps=self.BACKPROP_THROUGH_TIME_STEPS,
            optimization_algorithm_class=RNNGradientDescent,
            learning_rate=self.LEARNING_RATE,
            n_epochs=self.N_EPOCHS,
            parameter_initializer=OneOverRootNWeightInitializer,
            random_state=self.RANDOM_STATE
        )
        rnn_gradient_checker = RNNGradientChecker(network=network, X=self.X_TRAIN, y=self.Y_TRAIN)
        
        passes_gradient_check = rnn_gradient_checker.run()

        self.assertTrue(passes_gradient_check)
