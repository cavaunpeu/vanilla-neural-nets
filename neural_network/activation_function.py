from abc import ABCMeta, abstractmethod

import numpy as np


class BaseActivationFunction(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def activation_function(linear_combination):
        pass

    @abstractmethod
    def derivative_of_activation_function(linear_combination):
        pass


class SigmoidActivationFunction(BaseActivationFunction):

    @staticmethod
    def activation_function(linear_combination):
        return 1/(1 + np.exp(-linear_combination))

    @classmethod
    def derivative_of_activation_function(cls, linear_combination):
        return cls.activation_function(linear_combination) * (1 - cls.activation_function(linear_combination))


class SoftmaxActivationFunction(BaseActivationFunction):

    ARRAY_AXIS_TO_EXPAND = -2

    @classmethod
    def activation_function(cls, linear_combination):
        reshaped_linear_combination = np.expand_dims(linear_combination, axis=cls.ARRAY_AXIS_TO_EXPAND)
        return np.apply_along_axis(cls._softmax_function, axis=-1, arr=reshaped_linear_combination)\
            .squeeze(axis=cls.ARRAY_AXIS_TO_EXPAND)

    @classmethod
    def derivative_of_activation_function(cls, linear_combination):
        reshaped_linear_combination = np.expand_dims(linear_combination, axis=cls.ARRAY_AXIS_TO_EXPAND)
        n_training_instances = reshaped_linear_combination.shape[0]
        n_output_neurons = reshaped_linear_combination.shape[-1]

        return np.apply_along_axis(cls._derivative_of_softmax_function, axis=-1, arr=reshaped_linear_combination)\
            .reshape(n_training_instances, n_output_neurons, n_output_neurons)

    @classmethod
    def _derivative_of_softmax_function(cls, linear_combination):
        reshaped_linear_combination = np.expand_dims(linear_combination, axis=cls.ARRAY_AXIS_TO_EXPAND)
        diagonal_jacobian_entries = linear_combination * (1 - linear_combination)

        jacobian = reshaped_linear_combination.T.dot(reshaped_linear_combination)
        np.fill_diagonal(a=jacobian, val=diagonal_jacobian_entries)
        return jacobian.ravel()

    @staticmethod
    def _softmax_function(linear_combination):
        exponentiated_terms = np.exp(linear_combination)
        return exponentiated_terms / exponentiated_terms.sum()


class ReLUActivationFunction(BaseActivationFunction):

    @staticmethod
    def activation_function(linear_combination):
        return np.maximum(0, linear_combination)

    @classmethod
    def derivative_of_activation_function(cls, linear_combination):
        return np.vectorize(
            lambda linear_combination: 1 if linear_combination > 0 else 0
        )(linear_combination)


class LinearActivationFunction(BaseActivationFunction):

    @staticmethod
    def activation_function(linear_combination):
        return linear_combination

    @classmethod
    def derivative_of_activation_function(cls, linear_combination):
        return 1
