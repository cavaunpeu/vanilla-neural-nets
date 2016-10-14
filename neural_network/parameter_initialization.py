from base.parameter_initialization import BaseParameterInitializer


class GaussianWeightInitializer(BaseParameterInitializer):

    def __init__(self, standard_deviation, random_state):
        self.standard_deviation = standard_deviation
        super().__init__(random_state)

    def initialize(self, layer_sizes):
        return [
            self.random_number_generator.normal(loc=0, scale=self.standard_deviation, size=(next_layer_size, layer_size)) \
            for layer_size, next_layer_size \
            in zip(layer_sizes[:-1], layer_sizes[1:])
        ]


class GaussianBiasInitializer(BaseParameterInitializer):

    def __init__(self, standard_deviation, random_state):
        self.standard_deviation = standard_deviation
        super().__init__(random_state)

    def initialize(self, layer_sizes):
        return [
            self.random_number_generator.normal(loc=0, scale=self.standard_deviation, size=(layer_size,)) \
            for layer_size in layer_sizes[1:]
        ]
