import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from tests.helpers.gradient_check import GradientChecker
from vanilla_neural_nets.neural_network.network import VanillaNeuralNetwork
from vanilla_neural_nets.neural_network.training_batch_generator import MiniBatchGenerator
from vanilla_neural_nets.neural_network.optimization_algorithm import GradientDescent
from vanilla_neural_nets.neural_network.loss_function import MeanSquaredError, BinaryCrossEntropyLoss
from vanilla_neural_nets.neural_network.activation_function import SigmoidActivationFunction
from vanilla_neural_nets.neural_network.parameter_initialization import GaussianWeightInitializer, GaussianBiasInitializer


class TestVanillaNeuralNetwork(unittest.TestCase):

    N_EPOCHS = 1
    LEARNING_RATE = 1
    LAYER_SIZES = [4, 5, 6, 7, 8, 9, 2]
    RANDOM_STATE = 123
    GAUSSIAN_INITIALIZATER_STANDARD_DEVIATION = 1.

    X_TRAIN = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [5, 6, 5, 6], [7, 8, 7, 8]])
    Y_TRAIN = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    X_TEST = np.array([[7, 8, 7, 8], [9, 10, 9, 10]])

    TRAINING_BATCH_SIZE = len(X_TRAIN)

    MEAN_SQUARED_LOSS_EXPECTED_RESULT = np.array([[ 0.072563377, 0.4071061], [0.072517576, 0.407073701]])
    CROSS_ENTROPY_LOSS_EXPECTED_RESULT = np.array([[ 0.233455013, 0.477370927], [0.233337665, 0.477343897]])

    def test_network_with_mean_squared_error_loss_passes_gradient_check(self):
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
        gradient_checker = GradientChecker(network=network, X=self.X_TRAIN, y=self.Y_TRAIN)

        gradient_checker.run()

        self.assertTrue(gradient_checker.passed)


    def test_network_with_binary_cross_entropy_loss_passes_gradient_check(self):
        network = VanillaNeuralNetwork(
            layer_sizes=self.LAYER_SIZES,
            training_batch_generator_class=MiniBatchGenerator,
            loss_function_class=BinaryCrossEntropyLoss,
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
        gradient_checker = GradientChecker(network=network, X=self.X_TRAIN, y=self.Y_TRAIN)

        gradient_checker.run()

        self.assertTrue(gradient_checker.passed)


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

        network.fit(X=self.X_TRAIN, y=self.Y_TRAIN)

        actual_results = network.predict(x=self.X_TEST)
        assert_array_almost_equal(actual_results, self.MEAN_SQUARED_LOSS_EXPECTED_RESULT, decimal=9)


    def test_network_with_binary_cross_entropy_loss_gives_correct_output(self):
        network = VanillaNeuralNetwork(
            layer_sizes=self.LAYER_SIZES,
            training_batch_generator_class=MiniBatchGenerator,
            loss_function_class=BinaryCrossEntropyLoss,
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

        network.fit(X=self.X_TRAIN, y=self.Y_TRAIN)

        actual_results = network.predict(x=self.X_TEST)
        assert_array_almost_equal(actual_results, self.CROSS_ENTROPY_LOSS_EXPECTED_RESULT, decimal=9)
