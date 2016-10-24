from collections import deque

import numpy as np

from base.network import BaseRecurrentNeuralNetwork
from vanilla_neural_nets.recurrent_neural_network.parameter_object import RNNNetworkParametersCollection, LSTMNetworkParametersCollection


class VanillaRecurrentNeuralNetwork(BaseRecurrentNeuralNetwork):

    def _feed_forward(self, x):
        time_steps = len(x)
        initial_hidden_state = np.zeros(self.hidden_layer_size)
        hidden_state = deque([initial_hidden_state])
        softmax_outputs = deque()

        for t in np.arange(time_steps):
            hidden_state.append(
                np.tanh( self.parameters.W_xh.value[:, x[t]] + self.parameters.W_hh.value @ hidden_state[-1] )
            )
            softmax_outputs.append(
                self._compute_softmax( self.parameters.W_hy.value @ hidden_state[-1] )
            )
        # move initial hidden state to end of deque, such that it is later our
        # `hidden_state[t-1]` at t=0
        hidden_state.rotate(-1)

        return np.array(softmax_outputs), np.array(hidden_state)

    def _initialize_parameters(self):
        return RNNNetworkParametersCollection(
            vocabulary_size=self.vocabulary_size,
            hidden_layer_size=self.hidden_layer_size,
            weight_initializer=self.weight_initializer
        )


class VanillaLSTM(BaseRecurrentNeuralNetwork):

    def predict(self, x):
        softmax_outputs, hidden_state, cache = self._feed_forward(x)
        return softmax_outputs

    def _compute_training_loss(self, x, y_true):
        softmax_outputs, hidden_state, cache = self._feed_forward(x)
        return self.loss_function_class.total_loss(y_true=y_true, y_predicted=softmax_outputs)

    def _feed_forward(self, x):
        time_steps = len(x)
        initial_hidden_state = np.zeros(self.hidden_layer_size)
        hidden_state = deque([initial_hidden_state])
        softmax_outputs = deque()
        cache = self._create_empty_network_cache()

        for t in np.arange(time_steps):
            f_t = self._sigmoid(z=  self.parameters.W_fh.value @ hidden_state[-1] \
                                  + self.parameters.W_fx.value[:, x[t]] \
                                  + self.parameters.b_f.value)
            i_t = self._sigmoid(z=  self.parameters.W_ih.value @ hidden_state[-1] \
                                  + self.parameters.W_ix.value[:, x[t]] \
                                  + self.parameters.b_i.value)
            o_t = self._sigmoid(z=  self.parameters.W_oh.value @ hidden_state[-1] \
                                  + self.parameters.W_ox.value[:, x[t]] \
                                  + self.parameters.b_o.value)
            candidate_c_t = np.tanh(self.parameters.W_ch.value @ hidden_state[-1] \
                                  + self.parameters.W_cx.value[:, x[t]] \
                                  + self.parameters.b_c.value)
            c_t = f_t * cache['memory_cell'][-1] + i_t * candidate_c_t

            hidden_state.append( o_t * np.tanh(c_t) )
            softmax_outputs.append(
                self._compute_softmax( self.parameters.W_hy.value @ hidden_state[-1] + self.parameters.b_y.value )
            )
            cache['forget_gate'].append(f_t)
            cache['input_gate'].append(i_t)
            cache['output_gate'].append(o_t)
            cache['candidate_memory_cell'].append(candidate_c_t)
            cache['memory_cell'].append(c_t)

        # move initial hidden state and memory cell to end of deque, such that they are later our
        # `hidden_state[t-1]` and `cache['memory_cell']`, respectively, at t=0
        hidden_state.rotate(-1)
        cache['memory_cell'].rotate(-1)

        return np.array(softmax_outputs), np.array(hidden_state), cache

    def _initialize_parameters(self):
        return LSTMNetworkParametersCollection(
            vocabulary_size=self.vocabulary_size,
            hidden_layer_size=self.hidden_layer_size,
            weight_initializer=self.weight_initializer
        )

    def _create_empty_network_cache(self):
        initial_memory_cell = np.zeros(self.hidden_layer_size)
        return {
            'forget_gate': deque(),
            'input_gate': deque(),
            'output_gate': deque(),
            'candidate_memory_cell': deque(),
            'memory_cell': deque([initial_memory_cell])
        }

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))
