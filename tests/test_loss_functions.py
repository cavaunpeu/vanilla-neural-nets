import unittest

from vanilla_neural_nets.base.loss_function import BaseLossFunction
from vanilla_neural_nets.neural_network.loss_function import MeanSquaredError
from vanilla_neural_nets.neural_network.loss_function import BinaryCrossEntropyLoss
from vanilla_neural_nets.recurrent_neural_network.loss_function import CrossEntropyLoss


class TestLossFunctions(unittest.TestCase):

    def test_mean_squared_error_base_class_is_base_loss_function(self):
        mean_squared_error = MeanSquaredError()
        self.assertEqual(mean_squared_error.__class__.__base__, BaseLossFunction)

    def test_cross_entropy_base_class_is_base_loss_function(self):
        cross_entropy_loss = BinaryCrossEntropyLoss()
        self.assertEqual(cross_entropy_loss.__class__.__base__, BaseLossFunction)

    def test_rnn_cross_entropy_base_class_is_base_loss_function(self):
        rnn_cross_entropy_loss = CrossEntropyLoss()
        self.assertEqual(rnn_cross_entropy_loss.__class__.__base__, BaseLossFunction)
