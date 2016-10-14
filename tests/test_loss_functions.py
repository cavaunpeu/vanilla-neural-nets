import unittest

from base.loss_function import BaseLossFunction
from neural_network.loss_function import MeanSquaredError
from neural_network.loss_function import CrossEntropyLoss
from recurrent_neural_network.loss_function import CrossEntropyLoss as RNNCrossEntropyLoss


class TestLossFunctions(unittest.TestCase):

    def test_mean_squared_error_base_class_is_base_loss_function(self):
        mean_squared_error = MeanSquaredError()
        self.assertEqual(mean_squared_error.__class__.__base__, BaseLossFunction)

    def test_cross_entropy_base_class_is_base_loss_function(self):
        cross_entropy_loss = CrossEntropyLoss()
        self.assertEqual(cross_entropy_loss.__class__.__base__, BaseLossFunction)

    def test_rnn_cross_entropy_base_class_is_base_loss_function(self):
        # delete this test once you remove the cross-entropy loss from rnn module
        rnn_cross_entropy_loss = RNNCrossEntropyLoss()
        self.assertEqual(rnn_cross_entropy_loss.__class__.__base__, BaseLossFunction)
