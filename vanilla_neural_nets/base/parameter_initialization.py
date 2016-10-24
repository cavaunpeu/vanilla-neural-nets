from abc import ABCMeta, abstractmethod

import numpy as np


class BaseParameterInitializer(metaclass=ABCMeta):

    def __init__(self, random_state=123):
        self.random_number_generator = np.random.RandomState(random_state)

    @abstractmethod
    def initialize(layer_sizes):
        pass
