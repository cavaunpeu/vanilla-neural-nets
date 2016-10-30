from vanilla_neural_nets.base.parameter_initialization import BaseParameterInitializer



class GaussianWeightInitializer(BaseParameterInitializer):

    def __init__(self, standard_deviation, random_state):
        self.standard_deviation = standard_deviation
        super().__init__(random_state)

    def initialize(self, first_dimension, second_dimension):
        return self.random_number_generator.normal(loc=0, scale=self.standard_deviation, size=(first_dimension, second_dimension))


class GaussianBiasInitializer(BaseParameterInitializer):

    def __init__(self, standard_deviation, random_state):
        self.standard_deviation = standard_deviation
        super().__init__(random_state)

    def initialize(self, first_dimension):
        return self.random_number_generator.normal(loc=0, scale=self.standard_deviation, size=(first_dimension,))
