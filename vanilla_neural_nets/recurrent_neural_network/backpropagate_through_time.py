import numpy as np

from base.backpropagation import BaseBackPropagateThroughTime


class RNNBackPropagateThroughTime(BaseBackPropagateThroughTime):

    def compute_gradients(self, x, y):
        softmax_outputs, hidden_state = self.feed_forward_method(x)
        self.parameters.reset_gradients_to_zero()
        time_steps = np.arange(len(x))

        for t in time_steps[::-1]:
            label = y[t]

            # derivative of loss function w.r.t. softmax predictions
            dJdP = softmax_outputs[t]
            dJdP[label] -= 1

            # derivative of loss function w.r.t. W_hy
            self.parameters.W_hy.gradient += np.outer(dJdP, hidden_state[t])

            # derivative of loss function w.r.t. hidden state
            dJdH = dJdP @ self.parameters.W_hy.value

            # initialize dJdH_parent
            dJdH_parent = dJdH

            # back-propagate through time
            back_prop_through_time_steps = np.arange( max(0, t-self.backprop_through_time_steps), t+1)
            for t_ in back_prop_through_time_steps[::-1]:

                # derivative of loss function w.r.t. hidden-layer input
                dJdZ = dJdH_parent * (1 - hidden_state[t_]**2)

                # derivative of loss function w.r.t. W_hh
                self.parameters.W_hh.gradient += np.outer(dJdZ, hidden_state[t_-1])

                # derivative of loss function w.r.t. W_xh
                input_vector = np.arange(self.vocabulary_size) == x[t_]
                self.parameters.W_xh.gradient += np.outer(dJdZ, input_vector)

                # derivative of loss function w.r.t. *previous* hidden state
                dJdH_parent = dJdZ @ self.parameters.W_hh.value

        return self.parameters


class LSTMBackpropagateThroughTime(BaseBackPropagateThroughTime):

    def compute_gradients(self, x, y):
        softmax_outputs, hidden_state, cache = self.feed_forward_method(x)
        self.parameters.reset_gradients_to_zero()
        time_steps = np.arange(len(x))

        for t in time_steps[::-1]:
            label = y[t]

            # derivative of loss function w.r.t. softmax predictions
            dJdP = softmax_outputs[t]
            dJdP[label] -= 1

            # derivative of loss function w.r.t. W_hy
            self.parameters.W_hy.gradient += np.outer(dJdP, hidden_state[t])

            # derivative of loss function w.r.t. b_y
            self.parameters.b_y.gradient += dJdP

            # derivative of loss function w.r.t. hidden state
            dJdH = dJdP @ self.parameters.W_hy.value

            # initialize dJdH_parent
            dJdH_parent = dJdH

            # initialize dJdC_parent
            dJdC_parent = 0

            # back-propagate through time
            back_prop_through_time_steps = np.arange( max(0, t-self.backprop_through_time_steps), t+1)
            for t_ in back_prop_through_time_steps[::-1]:

                # build input vector
                input_vector = np.arange(self.vocabulary_size) == x[t_]

                # recompute tanh of memory cell, via h_t = o_t * tanh(c_t)
                tanh_memory_cell = hidden_state[t_] / cache['output_gate'][t_]

                # derivative of loss function w.r.t. o_t
                dJdO = dJdH_parent * tanh_memory_cell

                # derivative of loss function w.r.t. memory_cell
                dJdC = dJdC_parent + (dJdH_parent * cache['output_gate'][t_] * self._derivative_of_tanh(tanh_memory_cell))

                # derivative of loss function w.r.t. Z_{o_t}
                dJdZ_O = dJdO * self._derivative_of_sigmoid(cache['output_gate'][t_])

                # derivative of loss function w.r.t. Z_{candidate_c_t}
                dJdZ_C_candidate = dJdC * cache['input_gate'][t_] * self._derivative_of_tanh(cache['candidate_memory_cell'][t_])

                # derivative of loss function w.r.t. Z_{i_t}
                dJdZ_I = dJdC * cache['candidate_memory_cell'][t_] * self._derivative_of_sigmoid(cache['input_gate'][t_])

                # derivative of loss function w.r.t. Z_{f_t}
                dJdZ_F = dJdC * cache['memory_cell'][t_-1] * self._derivative_of_sigmoid(cache['forget_gate'][t_])

                # derivative of loss function w.r.t. W_oh
                self.parameters.W_oh.gradient += np.outer(dJdZ_O, hidden_state[t_-1])

                # derivative of loss function w.r.t. W_ox
                self.parameters.W_ox.gradient += np.outer(dJdZ_O, input_vector)

                # derivative of loss function w.r.t. b_o
                self.parameters.b_o.gradient += dJdZ_O

                # derivative of loss function w.r.t. W_ch
                self.parameters.W_ch.gradient += np.outer(dJdZ_C_candidate, hidden_state[t_-1])

                # derivative of loss function w.r.t. W_cx
                self.parameters.W_cx.gradient += np.outer(dJdZ_C_candidate, input_vector)

                # derivative of loss function w.r.t. b_c
                self.parameters.b_c.gradient += dJdZ_C_candidate

                # derivative of loss function w.r.t. W_ih
                self.parameters.W_ih.gradient += np.outer(dJdZ_I, hidden_state[t_-1])

                # derivative of loss function w.r.t. W_ix
                self.parameters.W_ix.gradient += np.outer(dJdZ_I, input_vector)

                # derivative of loss function w.r.t. b_i
                self.parameters.b_i.gradient += dJdZ_I

                # derivative of loss function w.r.t. W_fh
                self.parameters.W_fh.gradient += np.outer(dJdZ_F, hidden_state[t_-1])

                # derivative of loss function w.r.t. W_fx
                self.parameters.W_fx.gradient += np.outer(dJdZ_F, input_vector)

                # derivative of loss function w.r.t. b_f
                self.parameters.b_f.gradient += dJdZ_F

                # derivative of loss function w.r.t *previous* hidden-state, via o_t
                dJdH_parent_O = dJdZ_O @ self.parameters.W_oh.value

                # derivative of loss function w.r.t *previous* hidden-state, via candidate_c_t
                dJdH_parent_C_candidate = dJdZ_C_candidate @ self.parameters.W_ch.value

                # derivative of loss function w.r.t. *previous* hidden-state, via i_t
                dJdH_parent_I = dJdZ_I @ self.parameters.W_ih.value

                # derivative of loss function w.r.t. *previous* hidden-state, via f_t
                dJdH_parent_F = dJdZ_F @ self.parameters.W_fh.value

                # derivative of loss function w.r.t. *previous* hidden-state
                dJdH_parent = dJdH_parent_I + dJdH_parent_O + dJdH_parent_C_candidate + dJdH_parent_F

                # derivative of loss function w.r.t. *previous* memory cell, via current memory cell
                dJdC_parent = dJdC * cache['forget_gate'][t_]

        return self.parameters

    @staticmethod
    def _derivative_of_tanh(z):
        return 1 - z**2

    @staticmethod
    def _derivative_of_sigmoid(z):
        return z * (1 - z)
