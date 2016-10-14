from unittest.mock import patch
import sys

import numpy as np

from vanilla_neural_nets.recurrent_neural_network.optimization_algorithm import RNNGradientDescent


class RNNGradientChecker:

    def __init__(self, network, x, y, epsilon=.001, error_threshold=.01):
        self.network = network
        self.x = x
        self.y = y
        self.epsilon = epsilon
        self.error_threshold = error_threshold
        self.network_parameters = [
            self.network.parameters.W_xh,
            self.network.parameters.W_hh,
            self.network.parameters.W_hy
        ]
        self._passed = False

    @patch.object(RNNGradientDescent, '_update_weights')
    def run(self, mock_update_weights):
        self.network.fit(X=[self.x], y=[self.y])
        for parameter in self.network_parameters:
            if not self._passes_gradient_check(parameter=parameter):
                return
        self.passed = True

    @property
    def passed(self):
        return self._passed

    @passed.setter
    def passed(self, value):
        self._passed = value

    def _passes_gradient_check(self, parameter):
        iterator = np.nditer(parameter.value, flags=['multi_index'], op_flags=['readwrite'])

        while not iterator.finished:
            multi_index = iterator.multi_index
            numerical_gradient = self._compute_numerical_gradient(parameter=parameter, multi_index=multi_index)
            analytical_gradient = parameter.gradient[multi_index]

            relative_error = self._compute_relative_error(
                numerical_gradient=numerical_gradient,
                analytical_gradient=analytical_gradient
            )
            if relative_error > self.error_threshold:
                return False

            iterator.iternext()

        return True

    def _compute_numerical_gradient(self, parameter, multi_index):
        numerical_gradient = 0

        # f(x + h)
        tweaked_parameter_value = parameter.value.copy()
        tweaked_parameter_value[multi_index] = parameter.value[multi_index] + self.epsilon
        loss = self._compute_total_loss_given_parameter_value(parameter=parameter, value=tweaked_parameter_value)
        numerical_gradient += loss

        # - f(x - h)
        tweaked_parameter_value = parameter.value.copy()
        tweaked_parameter_value[multi_index] = parameter.value[multi_index] - self.epsilon
        loss = self._compute_total_loss_given_parameter_value(parameter=parameter, value=tweaked_parameter_value)
        numerical_gradient -= loss

        # /2h
        return numerical_gradient / (2 * self.epsilon)

    def _compute_total_loss_given_parameter_value(self, parameter, value):
        with patch.object(parameter, attribute='value', new=value):
            y_predicted = self.network.predict(x=self.x)
            return self.network.loss_function_class.total_loss(y_true=self.y, y_predicted=y_predicted)

    def _compute_relative_error(self, numerical_gradient, analytical_gradient):
        if numerical_gradient == analytical_gradient:
            return 0
        return np.abs( analytical_gradient - numerical_gradient ) / ( np.abs(analytical_gradient) + np.abs(numerical_gradient) )
