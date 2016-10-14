from base.training_batch_generator import BaseTrainingBatchGenerator
from neural_network.data_object import _TrainingBatch


class MiniBatchGenerator(BaseTrainingBatchGenerator):

    def __iter__(self):
        for start_index in self.batch_starting_indices:
            yield _TrainingBatch(
                X=self.X[start_index : start_index + self.batch_size],
                y=self.y[start_index : start_index + self.batch_size]
            )
