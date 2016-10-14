from abc import ABCMeta, abstractmethod


class BaseActivationFunction(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def activation_function(linear_combination):
        pass

    @abstractmethod
    def derivative_of_activation_function(linear_combination):
        pass
