import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from tests.fixtures.vanilla_neural_network_training_data_X import TRAINING_DATA_X
from tests.fixtures.vanilla_neural_network_training_data_y import TRAINING_DATA_Y
from tests.fixtures.vanilla_neural_network_test_data_X import TEST_DATA_X
from tests.fixtures.vanilla_neural_network_mean_squared_loss_expected_result import MEAN_SQUARED_LOSS_EXPECTED_RESULT
from tests.fixtures.vanilla_neural_network_cross_entropy_loss_expected_result import CROSS_ENTROPY_LOSS_EXPECTED_RESULT
from vanilla_neural_nets.neural_network.network import VanillaNeuralNetwork
from vanilla_neural_nets.neural_network.training_batch_generator import MiniBatchGenerator
from vanilla_neural_nets.neural_network.optimization_algorithm import GradientDescent
from vanilla_neural_nets.neural_network.loss_function import MeanSquaredError
from vanilla_neural_nets.neural_network.activation_function import SigmoidActivationFunction
from vanilla_neural_nets.neural_network.parameter_initialization import GaussianWeightInitializer, GaussianBiasInitializer


class TestVanillaNeuralNetwork(unittest.TestCase):

    N_EPOCHS = 1
    TRAINING_BATCH_SIZE = 10
    LEARNING_RATE = 3.
    LAYER_SIZES = [784, 30, 10]
    RANDOM_STATE = 123
    GAUSSIAN_INITIALIZATER_STANDARD_DEVIATION = 1.

    def test_network_with_mean_squared_loss_gives_correct_output(self):
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
            weight_initializer=GaussianWeightInitializer(
                self.GAUSSIAN_INITIALIZATER_STANDARD_DEVIATION,
                random_state=self.RANDOM_STATE
            ),
            bias_initializer=GaussianBiasInitializer(
                self.GAUSSIAN_INITIALIZATER_STANDARD_DEVIATION,
                random_state=self.RANDOM_STATE
            )
        )

        network.fit(X=TRAINING_DATA_X, y=TRAINING_DATA_Y)

        actual_results = network.predict(X=TEST_DATA_X)
        assert_array_almost_equal(actual_results, MEAN_SQUARED_LOSS_EXPECTED_RESULT, decimal=9)


    def test_network_with_cross_entropy_loss_gives_correct_output(self):
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
            weight_initializer=GaussianWeightInitializer(
                self.GAUSSIAN_INITIALIZATER_STANDARD_DEVIATION,
                random_state=self.RANDOM_STATE
            ),
            bias_initializer=GaussianBiasInitializer(
                self.GAUSSIAN_INITIALIZATER_STANDARD_DEVIATION,
                random_state=self.RANDOM_STATE
            )
        )

        network.fit(X=TRAINING_DATA_X, y=TRAINING_DATA_Y)

        actual_results = network.predict(X=TEST_DATA_X)
        assert_array_almost_equal(actual_results, CROSS_ENTROPY_LOSS_EXPECTED_RESULT, decimal=9)
