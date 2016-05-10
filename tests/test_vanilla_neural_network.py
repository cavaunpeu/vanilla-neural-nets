import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from tests.fixtures.vanilla_neural_network_training_data_X import TRAINING_DATA_X
from tests.fixtures.vanilla_neural_network_training_data_y import TRAINING_DATA_Y
from tests.fixtures.vanilla_neural_network_test_data_X import TEST_DATA_X
from tests.fixtures.vanilla_neural_network_expected_result import EXPECTED_RESULT
from neural_network.network import VanillaNeuralNetwork
from neural_network.training_batch_generator import MiniBatchGenerator
from neural_network.optimization_algorithm import GradientDescent
from neural_network.loss_function import MeanSquaredError
from neural_network.activation_function import SigmoidActivationFunction


class TestVanillaNeuralNetwork(unittest.TestCase):

    N_EPOCHS = 1
    TRAINING_BATCH_SIZE = 10
    LEARNING_RATE = 3.
    LAYER_SIZES = [784, 30, 10]
    RANDOM_STATE = 123

    def test_network_gives_correct_output(self):
        network = VanillaNeuralNetwork(
            layer_sizes=self.LAYER_SIZES,
            training_batch_generator_class=MiniBatchGenerator,
            loss_function_class=MeanSquaredError,
            activation_function_class=SigmoidActivationFunction,
            optimization_algorithm_class=GradientDescent,
            learning_rate=self.LEARNING_RATE,
            n_epochs=self.N_EPOCHS,
            training_batch_size=self.TRAINING_BATCH_SIZE,
            random_state=self.RANDOM_STATE,
        )

        network.fit(X=TRAINING_DATA_X, y=TRAINING_DATA_Y)

        actual_results = network.predict(X=TEST_DATA_X)
        assert_array_almost_equal(actual_results, EXPECTED_RESULT, decimal=9)
