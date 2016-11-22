from vanilla_neural_nets.base.parameter_object import _NetworkWeightParameter, _NetworkBiasParameter


class _NetworkLayer:

    def __init__(self, weight_parameter, bias_parameter, is_output_layer):
        self.weight_parameter = weight_parameter
        self.bias_parameter = bias_parameter
        self.is_output_layer = is_output_layer


class NetworkLayersCollection:

    def __init__(self, layer_sizes, weight_initializer, bias_initializer):
        self.layer_sizes = layer_sizes
        self.weight_parameters = self._initialize_weight_parameters(weight_initializer)
        self.bias_parameters = self._initialize_bias_parameters(bias_initializer)

    def reset_gradients_to_zero(self):
        for parameter in self._parameters:
            parameter.reset_gradient_to_zero()

    def _initialize_weight_parameters(self, weight_initializer):
        weight_parameters = []
        for layer_index, (layer_size, next_layer_size) in enumerate( zip(self.layer_sizes[:-1], self.layer_sizes[1:]) ):
            weight_parameters.append(
                _NetworkWeightParameter(
                    name='W_' + str(layer_index),
                    first_dimension=next_layer_size,
                    second_dimension=layer_size,
                    weight_initializer=weight_initializer
                )
            )
        return weight_parameters

    def _initialize_bias_parameters(self, bias_initializer):
        bias_parameters = []
        for layer_index, layer_size in enumerate(self.layer_sizes[1:]):
            bias_parameters.append(
                _NetworkBiasParameter(
                    name='b_' + str(layer_index),
                    first_dimension=layer_size,
                    bias_initializer=bias_initializer
                )
            )
        return bias_parameters

    @property
    def layers(self):
        for layer_index, (weight_parameter, bias_parameter) in enumerate(zip(self.weight_parameters, self.bias_parameters)):
            is_output_layer = layer_index + 1 == len(self.weight_parameters)
            yield _NetworkLayer(
                weight_parameter=weight_parameter,
                bias_parameter=bias_parameter,
                is_output_layer=is_output_layer
            )

    @property
    def _parameters(self):
        return self.weight_parameters + self.bias_parameters

    def __iter__(self):
        for parameter in self._parameters:
            yield parameter
