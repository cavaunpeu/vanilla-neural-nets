from vanilla_neural_nets.base.optimization_algorithm import BaseRNNGradientDescent


class RNNGradientDescent(BaseRNNGradientDescent):

    def _update_weights(self):
        self.parameters.W_xh.value -= self.learning_rate * self.parameters.W_xh.gradient
        self.parameters.W_hh.value -= self.learning_rate * self.parameters.W_hh.gradient
        self.parameters.W_hy.value -= self.learning_rate * self.parameters.W_hy.gradient


class LSTMGradientDescent(BaseRNNGradientDescent):

    def _update_weights(self):
        self.parameters.W_fh.value -= self.learning_rate * self.parameters.W_fh.gradient
        self.parameters.W_fx.value -= self.learning_rate * self.parameters.W_fx.gradient
        self.parameters.W_ih.value -= self.learning_rate * self.parameters.W_ih.gradient
        self.parameters.W_ix.value -= self.learning_rate * self.parameters.W_ix.gradient
        self.parameters.W_oh.value -= self.learning_rate * self.parameters.W_oh.gradient
        self.parameters.W_ox.value -= self.learning_rate * self.parameters.W_ox.gradient
        self.parameters.W_ch.value -= self.learning_rate * self.parameters.W_ch.gradient
        self.parameters.W_cx.value -= self.learning_rate * self.parameters.W_cx.gradient
        self.parameters.W_hy.value -= self.learning_rate * self.parameters.W_hy.gradient
        self.parameters.b_f .value-= self.learning_rate * self.parameters.b_f.gradient
        self.parameters.b_i .value-= self.learning_rate * self.parameters.b_i.gradient
        self.parameters.b_o .value-= self.learning_rate * self.parameters.b_o.gradient
        self.parameters.b_c .value-= self.learning_rate * self.parameters.b_c.gradient
        self.parameters.b_y .value-= self.learning_rate * self.parameters.b_y.gradient
