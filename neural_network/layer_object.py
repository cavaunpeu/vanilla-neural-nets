
class _NetworkLayer:

    def __init__(self, weight_matrix, bias_vector, output_layer):
        self.weight_matrix = weight_matrix
        self.bias_vector = bias_vector
        self.output_layer = output_layer


class NetworkLayersCollection:

    def __init__(self, layer_sizes, weight_initializer, bias_initializer):
        self._weight_matrices = weight_initializer.initialize(layer_sizes)
        self._bias_vectors = bias_initializer.initialize(layer_sizes)

    @property
    def bias_vectors(self):
        return self._bias_vectors

    @property
    def weight_matrices(self):
        return self._weight_matrices

    @bias_vectors.setter
    def bias_vectors(self, updated_bias_vectors):
        self._bias_vectors = updated_bias_vectors

    @weight_matrices.setter
    def weight_matrices(self, updated_weight_matrices):
        self._weight_matrices = updated_weight_matrices

    def __iter__(self):
        for layer_index, (weight_matrix, bias_vector) in enumerate(zip(self.weight_matrices, self.bias_vectors)):
            is_output_layer = layer_index + 1 == len(self.weight_matrices)
            yield _NetworkLayer(
                weight_matrix=weight_matrix,
                bias_vector=bias_vector,
                output_layer=is_output_layer
            )
