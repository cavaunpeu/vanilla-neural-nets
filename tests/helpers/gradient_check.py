from unittest.mock import patch

import numpy as np


class GradientChecker:

    def __init__(self, network, X, y, epsilon=.0001, error_threshold=.01):
        self.network = network
        self.X = X
        self.y = y
        self.epsilon = epsilon
        self.error_threshold = error_threshold
        self._passed = False

    def run(self):
        with patch.object(self.network.optimization_algorithm_class, attribute='_update_parameters'):
            self.network.fit(X=self.X, y=self.y)
            for parameter in self.network.parameters:
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
            if (relative_error > self.error_threshold) or np.isnan(relative_error):
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
            X, y = self._normalize_X_and_y_for_prediction()
            y_predicted = self.network.predict(X)
            return self.network.loss_function_class.total_loss(y_true=y, y_predicted=y_predicted)

    def _compute_relative_error(self, numerical_gradient, analytical_gradient):
        if numerical_gradient == analytical_gradient:
            return 0
        return np.abs( analytical_gradient - numerical_gradient ) / ( np.abs(analytical_gradient) + np.abs(numerical_gradient) )

    def _normalize_X_and_y_for_prediction(self):
        """The RNN `predict` expects a single list. The neural net `predict` expects
        a matrix. This is not ideal. We leave it for now.
        """
        return np.squeeze(self.X), np.squeeze(self.y)


class SparsityEnforcingGradientChecker(GradientChecker):

    def _compute_total_loss_given_parameter_value(self, parameter, value):
        with patch.object(parameter, attribute='value', new=value):
            y_predicted, hidden_layer_sparsity_coefficients = self.network.predict(self.X)
            return self.network.loss_function_class.total_loss(y_true=self.y, y_predicted=y_predicted, rho=self.network.rho, beta=self.network.beta, vector_of_rho_hats=hidden_layer_sparsity_coefficients)
