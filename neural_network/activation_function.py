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
