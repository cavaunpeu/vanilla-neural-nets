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
        activations = cls.activation_function(linear_combination)
        return activations * (1 - activations)

    @staticmethod
    def _softmax_function(linear_combination):
        normalized_linear_combination = linear_combination - linear_combination.max()
        exponentiated_terms = np.exp(normalized_linear_combination)
        return exponentiated_terms / exponentiated_terms.sum()


class ReLUActivationFunction(BaseActivationFunction):

    @staticmethod
    def activation_function(linear_combination):
        return np.maximum(0, linear_combination)

    @classmethod
    def derivative_of_activation_function(cls, linear_combination):
        return linear_combination > 0


class LinearActivationFunction(BaseActivationFunction):

    @staticmethod
    def activation_function(linear_combination):
        return linear_combination

    @classmethod
    def derivative_of_activation_function(cls, linear_combination):
        return 1
