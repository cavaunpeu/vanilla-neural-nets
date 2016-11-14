import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from tests.helpers.gradient_check import GradientChecker
from vanilla_neural_nets.autoencoder.sparse_autoencoder.network import VanillaSparseAutoencoder
from vanilla_neural_nets.neural_network.training_batch_generator import MiniBatchGenerator
from vanilla_neural_nets.neural_network.optimization_algorithm import GradientDescent
from vanilla_neural_nets.autoencoder.sparse_autoencoder.loss_function import SparseMeanSquaredError
from vanilla_neural_nets.neural_network.activation_function import SigmoidActivationFunction
from vanilla_neural_nets.neural_network.parameter_initialization import GaussianWeightInitializer, GaussianBiasInitializer


class TestSparseAutoencoder(unittest.TestCase):

    LAYER_SIZES = [10, 5, 10]
    LEARNING_RATE = 1
    N_EPOCHS = 1
    TRAINING_BATCH_SIZE = 1
    RANDOM_STATE = 123
    GAUSSIAN_INITIALIZATER_STANDARD_DEVIATION = 1.

    X_TRAIN = np.array([
        [3, 4, 0, 1, 1, 1, 2, 3, 0, 1],
        [1, 1, 0, 0, 1, 2, 1, 3, 4, 0],
        [1, 1, 3, 2, 1, 2, 4, 1, 1, 4]
    ])
    Y_TRAIN = X_TRAIN

    def test_sparse_autoencoder_with_sparse_mean_squared_error_loss_passes_gradient_check(self):
            network = VanillaSparseAutoencoder(
                layer_sizes=self.LAYER_SIZES,
                training_batch_generator_class=MiniBatchGenerator,
                loss_function_class=SparseMeanSquaredError,
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
