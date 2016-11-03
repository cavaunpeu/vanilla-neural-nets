import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from vanilla_neural_nets.recurrent_neural_network.network import VanillaRecurrentNeuralNetwork, VanillaLSTM
from vanilla_neural_nets.recurrent_neural_network.optimization_algorithm import RNNGradientDescent, LSTMGradientDescent
from vanilla_neural_nets.recurrent_neural_network.backpropagate_through_time import RNNBackPropagateThroughTime, LSTMBackpropagateThroughTime
from vanilla_neural_nets.recurrent_neural_network.parameter_initialization import OneOverRootNWeightInitializer
from tests.helpers.gradient_check import GradientChecker


class TestVanillaRecurrentNeuralNetwork(unittest.TestCase):

    X_TRAIN = [[0, 1, 2, 3]]
    Y_TRAIN = [[1, 2, 3, 4]]
    VOCABULARY_SIZE = 10

    HIDDEN_LAYER_SIZE = 3
    BACKPROP_THROUGH_TIME_STEPS = SOME_LARGE_NUMBER = 1000
    LEARNING_RATE = .005
    N_EPOCHS = 1
    RANDOM_STATE = 12345

    EXPECTED_W_XH_VALUE_AFTER_GRADIENT_DESCENT_STEP = np.array([
       [ 0.27266396, -0.11674642, -0.19921587, -0.18784354,  0.04283307,
         0.06042778,  0.29378478,  0.0968777 ,  0.15742238,  0.09712611],
       [ 0.1556133 ,  0.29201676, -0.312071  , -0.24786636, -0.12731095,
         0.09892312,  0.19594266,  0.23538472,  0.29386894,  0.14147104],
       [ 0.08855106,  0.13706159, -0.02076407, -0.10934516, -0.0381721 ,
         0.14526813,  0.31244226,  0.11186476,  0.18393231, -0.2081321 ]
    ])
    EXPECTED_W_HH_VALUE_AFTER_GRADIENT_DESCENT_STEP = np.array([
       [-0.29942427,  0.19027921,  0.25542752],
       [-0.30047605, -0.005691  ,  0.01641992],
       [ 0.06078078, -0.28374841,  0.24974508]
    ])
    EXPECTED_W_HY_VALUE_AFTER_GRADIENT_DESCENT_STEP = np.array([
       [  1.44439933e-01,   2.01375000e-01,   7.73200095e-05],
       [  1.97596028e-01,  -2.54676996e-01,  -1.77348405e-01],
       [ -1.53207158e-01,  -1.90581668e-02,  -2.50902918e-02],
       [  1.32175349e-01,  -2.04852597e-01,   1.95622959e-02],
       [ -2.11150986e-01,   1.68954858e-01,   2.70459034e-01],
       [  6.93445299e-02,  -2.21168857e-01,  -6.60900394e-03],
       [ -7.74892307e-02,   2.20505673e-01,   2.59934639e-01],
       [ -7.33604179e-02,  -1.16621514e-01,   4.32052142e-02],
       [ -1.97327857e-01,  -2.36555524e-01,   1.18599250e-01],
       [  1.89559120e-01,   4.65433020e-02,   2.99233326e-01]
    ])

    X_TEST = [1, 3, 5, 7]
    EXPECTED_PREDICTIONS = np.array([
       [ 0.1039706 ,  0.08861039,  0.10075295,  0.09302079,  0.11139207,
         0.09293745,  0.11113309,  0.09800194,  0.09708666,  0.10309406],
       [ 0.09527139,  0.10748862,  0.10210434,  0.1034768 ,  0.09393808,
         0.10469148,  0.09229414,  0.10253984,  0.10470121,  0.0934941 ],
       [ 0.10145951,  0.09353862,  0.09854082,  0.09708348,  0.10557278,
         0.09650587,  0.10600911,  0.09848255,  0.09830849,  0.10449877],
       [ 0.10707651,  0.09506989,  0.09672738,  0.09742555,  0.10369424,
         0.09579618,  0.1070661 ,  0.09653   ,  0.09285929,  0.10775486]
    ])

    def test_network_passes_gradient_check(self):
        network = VanillaRecurrentNeuralNetwork(
            vocabulary_size=self.VOCABULARY_SIZE,
            hidden_layer_size=self.HIDDEN_LAYER_SIZE,
            backprop_through_time_class=RNNBackPropagateThroughTime,
            backprop_through_time_steps=self.BACKPROP_THROUGH_TIME_STEPS,
            optimization_algorithm_class=RNNGradientDescent,
            weight_initializer_class=OneOverRootNWeightInitializer,
            learning_rate=self.LEARNING_RATE,
            n_epochs=self.N_EPOCHS,
            random_state=self.RANDOM_STATE
        )
        gradient_checker = GradientChecker(network=network, X=self.X_TRAIN, y=self.Y_TRAIN)

        gradient_checker.run()

        self.assertTrue(gradient_checker.passed)

    def test_network_correctly_updates_W_xh_value_after_gradient_descent_step(self):
        network = VanillaRecurrentNeuralNetwork(
            vocabulary_size=self.VOCABULARY_SIZE,
            hidden_layer_size=self.HIDDEN_LAYER_SIZE,
            backprop_through_time_class=RNNBackPropagateThroughTime,
            backprop_through_time_steps=self.BACKPROP_THROUGH_TIME_STEPS,
            optimization_algorithm_class=RNNGradientDescent,
            weight_initializer_class=OneOverRootNWeightInitializer,
            learning_rate=self.LEARNING_RATE,
            n_epochs=self.N_EPOCHS,
            random_state=self.RANDOM_STATE
        )

        network.fit(X=self.X_TRAIN, y=self.Y_TRAIN)

        assert_array_almost_equal(network.parameters.W_xh.value,
            self.EXPECTED_W_XH_VALUE_AFTER_GRADIENT_DESCENT_STEP, decimal=8)

    def test_network_correctly_updates_W_hh_value_after_gradient_descent_step(self):
        network = VanillaRecurrentNeuralNetwork(
            vocabulary_size=self.VOCABULARY_SIZE,
            hidden_layer_size=self.HIDDEN_LAYER_SIZE,
            backprop_through_time_class=RNNBackPropagateThroughTime,
            backprop_through_time_steps=self.BACKPROP_THROUGH_TIME_STEPS,
            optimization_algorithm_class=RNNGradientDescent,
            weight_initializer_class=OneOverRootNWeightInitializer,
            learning_rate=self.LEARNING_RATE,
            n_epochs=self.N_EPOCHS,
            random_state=self.RANDOM_STATE
        )

        network.fit(X=self.X_TRAIN, y=self.Y_TRAIN)

        assert_array_almost_equal(network.parameters.W_hh.value,
            self.EXPECTED_W_HH_VALUE_AFTER_GRADIENT_DESCENT_STEP, decimal=8)

    def test_network_correctly_updates_W_hy_value_after_gradient_descent_step(self):
        network = VanillaRecurrentNeuralNetwork(
            vocabulary_size=self.VOCABULARY_SIZE,
            hidden_layer_size=self.HIDDEN_LAYER_SIZE,
            backprop_through_time_class=RNNBackPropagateThroughTime,
            backprop_through_time_steps=self.BACKPROP_THROUGH_TIME_STEPS,
            optimization_algorithm_class=RNNGradientDescent,
            weight_initializer_class=OneOverRootNWeightInitializer,
            learning_rate=self.LEARNING_RATE,
            n_epochs=self.N_EPOCHS,
            random_state=self.RANDOM_STATE
        )

        network.fit(X=self.X_TRAIN, y=self.Y_TRAIN)

        assert_array_almost_equal(network.parameters.W_hy.value,
            self.EXPECTED_W_HY_VALUE_AFTER_GRADIENT_DESCENT_STEP, decimal=8)

    def test_network_makes_correct_predictions(self):
        network = VanillaRecurrentNeuralNetwork(
            vocabulary_size=self.VOCABULARY_SIZE,
            hidden_layer_size=self.HIDDEN_LAYER_SIZE,
            backprop_through_time_class=RNNBackPropagateThroughTime,
            backprop_through_time_steps=self.BACKPROP_THROUGH_TIME_STEPS,
            optimization_algorithm_class=RNNGradientDescent,
            weight_initializer_class=OneOverRootNWeightInitializer,
            learning_rate=self.LEARNING_RATE,
            n_epochs=self.N_EPOCHS,
            random_state=self.RANDOM_STATE
        )

        network.fit(X=self.X_TRAIN, y=self.Y_TRAIN)
        predictions = network.predict(self.X_TEST)

        assert_array_almost_equal(predictions, self.EXPECTED_PREDICTIONS, decimal=8)


class TestLSTM(unittest.TestCase):

    X_TRAIN = [[0, 1, 2, 3]]
    Y_TRAIN = [[1, 2, 3, 4]]
    VOCABULARY_SIZE = 10
    HIDDEN_LAYER_SIZE = 3
    BACKPROP_THROUGH_TIME_STEPS = SOME_LARGE_NUMBER = 1000
    LEARNING_RATE = .005
    N_EPOCHS = 1
    RANDOM_STATE = 12345

    def test_network_passes_gradient_check(self):
        network = VanillaLSTM(
            vocabulary_size=self.VOCABULARY_SIZE,
            hidden_layer_size=self.HIDDEN_LAYER_SIZE,
            backprop_through_time_class=LSTMBackpropagateThroughTime,
            backprop_through_time_steps=self.BACKPROP_THROUGH_TIME_STEPS,
            optimization_algorithm_class=LSTMGradientDescent,
            weight_initializer_class=OneOverRootNWeightInitializer,
            learning_rate=self.LEARNING_RATE,
            n_epochs=self.N_EPOCHS,
            random_state=self.RANDOM_STATE
        )
        gradient_checker = GradientChecker(network=network, X=self.X_TRAIN, y=self.Y_TRAIN)

        gradient_checker.run()

        self.assertTrue(gradient_checker.passed)
