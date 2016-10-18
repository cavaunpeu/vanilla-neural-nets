import numpy as np


class _NetworkWeightParameter:

    def __init__(self, name, first_dimension, second_dimension, weight_initializer):
        self.name = name
        self.value = weight_initializer.initialize(
            first_dimension=first_dimension,
            second_dimension=second_dimension
        )
        self.reset_gradient_to_zero()

    def reset_gradient_to_zero(self):
        self.gradient = np.zeros_like(self.value)


class _NetworkBiasParameter:

    def __init__(self, name, size):
        self.name = name
        self.value = np.zeros(size)
        self.reset_gradient_to_zero()

    def reset_gradient_to_zero(self):
        self.gradient = np.zeros_like(self.value)


class RNNNetworkParametersCollection:

    def __init__(self, vocabulary_size, hidden_layer_size, weight_initializer):
        self.W_xh = _NetworkWeightParameter(
            name='W_xh',
            first_dimension=hidden_layer_size,
            second_dimension=vocabulary_size,
            weight_initializer=weight_initializer
        )
        self.W_hh = _NetworkWeightParameter(
            name='W_hh',
            first_dimension=hidden_layer_size,
            second_dimension=hidden_layer_size,
            weight_initializer=weight_initializer
        )
        self.W_hy = _NetworkWeightParameter(
            name='W_hy',
            first_dimension=vocabulary_size,
            second_dimension=hidden_layer_size,
            weight_initializer=weight_initializer
        )

    def reset_gradients_to_zero(self):
        self.W_xh.reset_gradient_to_zero()
        self.W_hh.reset_gradient_to_zero()
        self.W_hy.reset_gradient_to_zero()

    @property
    def _parameters(self):
        return [self.W_xh, self.W_hh, self.W_hy]

    def __iter__(self):
        for parameter in self._parameters:
            yield parameter


class LSTMNetworkParametersCollection:

    def __init__(self, vocabulary_size, hidden_layer_size, weight_initializer):
        self.W_fh = _NetworkWeightParameter(
            name='W_fh',
            first_dimension=hidden_layer_size,
            second_dimension=hidden_layer_size,
            weight_initializer=weight_initializer
        )
        self.W_fx = _NetworkWeightParameter(
            name='W_fx',
            first_dimension=hidden_layer_size,
            second_dimension=vocabulary_size,
            weight_initializer=weight_initializer
        )
        self.W_ih = _NetworkWeightParameter(
            name='W_ih',
            first_dimension=hidden_layer_size,
            second_dimension=hidden_layer_size,
            weight_initializer=weight_initializer
        )
        self.W_ix = _NetworkWeightParameter(
            name='W_ix',
            first_dimension=hidden_layer_size,
            second_dimension=vocabulary_size,
            weight_initializer=weight_initializer
        )
        self.W_oh = _NetworkWeightParameter(
            name='W_oh',
            first_dimension=hidden_layer_size,
            second_dimension=hidden_layer_size,
            weight_initializer=weight_initializer
        )
        self.W_ox = _NetworkWeightParameter(
            name='W_ox',
            first_dimension=hidden_layer_size,
            second_dimension=vocabulary_size,
            weight_initializer=weight_initializer
        )
        self.W_ch = _NetworkWeightParameter(
            name='W_ch',
            first_dimension=hidden_layer_size,
            second_dimension=hidden_layer_size,
            weight_initializer=weight_initializer
        )
        self.W_cx = _NetworkWeightParameter(
            name='W_cx',
            first_dimension=hidden_layer_size,
            second_dimension=vocabulary_size,
            weight_initializer=weight_initializer
        )
        self.W_hy = _NetworkWeightParameter(
            name='W_hy',
            first_dimension=vocabulary_size,
            second_dimension=hidden_layer_size,
            weight_initializer=weight_initializer
        )
        self.b_f = _NetworkBiasParameter(name='b_f', size=hidden_layer_size)
        self.b_i = _NetworkBiasParameter(name='b_f', size=hidden_layer_size)
        self.b_o = _NetworkBiasParameter(name='b_f', size=hidden_layer_size)
        self.b_c = _NetworkBiasParameter(name='b_f', size=hidden_layer_size)
        self.b_y = _NetworkBiasParameter(name='b_f', size=vocabulary_size)

    def reset_gradients_to_zero(self):
        self.W_fh.reset_gradient_to_zero()
        self.W_fx.reset_gradient_to_zero()
        self.W_ih.reset_gradient_to_zero()
        self.W_ix.reset_gradient_to_zero()
        self.W_oh.reset_gradient_to_zero()
        self.W_ox.reset_gradient_to_zero()
        self.W_ch.reset_gradient_to_zero()
        self.W_cx.reset_gradient_to_zero()
        self.W_hy.reset_gradient_to_zero()
        self.b_f.reset_gradient_to_zero()
        self.b_i.reset_gradient_to_zero()
        self.b_o.reset_gradient_to_zero()
        self.b_c.reset_gradient_to_zero()
        self.b_y.reset_gradient_to_zero()

    @property
    def _parameters(self):
        return [
            self.W_fh,
            self.W_fx,
            self.W_ih,
            self.W_ix,
            self.W_oh,
            self.W_ox,
            self.W_ch,
            self.W_cx,
            self.W_hy,
            self.b_f,
            self.b_i,
            self.b_o,
            self.b_c,
            self.b_y,
        ]

    def __iter__(self):
        for parameter in self._parameters:
            yield parameter
