import numpy as np

from vanilla_neural_nets.base.parameter_initialization import BaseParameterInitializer


class OneOverRootNWeightInitializer(BaseParameterInitializer):

    def __init__(self, vocabulary_size, random_state):
        self.vocabulary_size = vocabulary_size
        super().__init__(random_state)

    def initialize(self, first_dimension, second_dimension):
        return self.random_number_generator.uniform(
            low=-np.sqrt(1./self.vocabulary_size),
            high=np.sqrt(1./self.vocabulary_size),
            size=(first_dimension, second_dimension)
        )
