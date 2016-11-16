import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from tests.helpers.gradient_check import SparsityEnforcingGradientChecker
from vanilla_neural_nets.autoencoder.sparse_autoencoder.network import VanillaSparseAutoencoder
from vanilla_neural_nets.neural_network.training_batch_generator import MiniBatchGenerator
from vanilla_neural_nets.autoencoder.sparse_autoencoder.optimization_algorithm import SparsityEnforcingGradientDescent
from vanilla_neural_nets.autoencoder.sparse_autoencoder.loss_function import SparseMeanSquaredError, KLDivergenceSparsityLoss
from vanilla_neural_nets.neural_network.activation_function import SigmoidActivationFunction
from vanilla_neural_nets.neural_network.parameter_initialization import GaussianWeightInitializer, GaussianBiasInitializer


class TestSparseAutoencoder(unittest.TestCase):

    LAYER_SIZES = [3, 2, 3]
    LEARNING_RATE = 1
    N_EPOCHS = 1
    RANDOM_STATE = 123
    GAUSSIAN_INITIALIZATER_STANDARD_DEVIATION = 1.
    RHO = .5
    BETA = .05

    X_TRAIN = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    Y_TRAIN = X_TRAIN

    TRAINING_BATCH_SIZE = len(X_TRAIN)

    def test_sparse_autoencoder_with_sparse_mean_squared_error_loss_passes_gradient_check(self):
        network = VanillaSparseAutoencoder(
            layer_sizes=self.LAYER_SIZES,
            training_batch_generator_class=MiniBatchGenerator,
            loss_function_class=SparseMeanSquaredError,
            activation_function_class=SigmoidActivationFunction,
            optimization_algorithm_class=SparsityEnforcingGradientDescent,
            sparsity_constraint_class=KLDivergenceSparsityLoss,
            learning_rate=self.LEARNING_RATE,
            n_epochs=self.N_EPOCHS,
            training_batch_size=self.TRAINING_BATCH_SIZE,
            random_state=self.RANDOM_STATE,
            rho=self.RHO,
            beta=self.BETA,
            weight_initializer=GaussianWeightInitializer(
                self.GAUSSIAN_INITIALIZATER_STANDARD_DEVIATION,
                random_state=self.RANDOM_STATE
            ),
            bias_initializer=GaussianBiasInitializer(
                self.GAUSSIAN_INITIALIZATER_STANDARD_DEVIATION,
                random_state=self.RANDOM_STATE
            )
        )
        gradient_checker = SparsityEnforcingGradientChecker(network=network, X=self.X_TRAIN, y=self.Y_TRAIN)

        gradient_checker.run()

        self.assertTrue(gradient_checker.passed)
