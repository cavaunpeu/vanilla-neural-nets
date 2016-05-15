
class NetworkLayer:

    def __init__(self, weight_matrix, bias_vector, output_layer):
        self.weight_matrix = weight_matrix
        self.bias_vector = bias_vector
        self.output_layer = output_layer


class NetworkLayersCollection:

    def __init__(self, layer_sizes, random_number_generator):
        self.layer_sizes = layer_sizes
        self.random_number_generator = random_number_generator
        self._bias_vectors = self._initialize_bias_vectors()
        self._weight_matrices = self._initialize_weight_matrices()

    def _initialize_weight_matrices(self):
        return [self.random_number_generator.randn(next_layer_size, layer_size) for layer_size, next_layer_size \
            in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def _initialize_bias_vectors(self):
        return [self.random_number_generator.randn(layer_size) for layer_size in self.layer_sizes[1:]]

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
        for layer_number, (weight_matrix, bias_vector) in enumerate(zip(self.weight_matrices, self.bias_vectors)):
            yield NetworkLayer(
                weight_matrix=weight_matrix, 
                bias_vector=bias_vector,
                output_layer=layer_number == len(self.layer_sizes)
            )