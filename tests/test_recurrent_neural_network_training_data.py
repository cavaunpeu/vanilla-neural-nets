import unittest

from recurrent_neural_network.training_data import WordLevelRNNTrainingDataBuilder


class TestRecurrentNeuralNetworkTrainingData(unittest.TestCase):

    CORPUS = 'the dog went to the store. it then chased me home.'

    VOCABULARY_SIZE_WHEREBY_NO_WORDS_REMOVED = SOME_LARGE_NUMBER = 100
    EXPECTED_X_TRAIN_WHEN_NO_WORDS_REMOVED = [[2, 9, 4, 12, 11, 9, 8, 0], [2, 6, 10, 3, 7, 5, 0]]
    EXPECTED_Y_TRAIN_WHEN_NO_WORDS_REMOVED = [[9, 4, 12, 11, 9, 8, 0, 1], [6, 10, 3, 7, 5, 0, 1]]

    VOCABULARY_SIZE_WHEREBY_SOME_WORDS_REMOVED = 4
    EXPECTED_X_TRAIN_WHEN_SOME_WORDS_REMOVED = [[2, 4, 3, 3, 3, 4, 3, 0], [2, 3, 3, 3, 3, 3, 0]]
    EXPECTED_Y_TRAIN_WHEN_SOME_WORDS_REMOVED = [[4, 3, 3, 3, 4, 3, 0, 1], [3, 3, 3, 3, 3, 0, 1]]

    def test_X_train_correctly_encoded_as_indices_when_no_words_removed(self):
        training_data = WordLevelRNNTrainingDataBuilder.build(
            corpus=self.CORPUS,
            vocabulary_size=self.VOCABULARY_SIZE_WHEREBY_NO_WORDS_REMOVED
        )

        self.assertEqual(training_data.X_train,
            self.EXPECTED_X_TRAIN_WHEN_NO_WORDS_REMOVED)

    def test_y_train_correctly_encoded_as_indices_when_no_words_removed(self):
        training_data = WordLevelRNNTrainingDataBuilder.build(
            corpus=self.CORPUS,
            vocabulary_size=self.VOCABULARY_SIZE_WHEREBY_NO_WORDS_REMOVED
        )

        self.assertEqual(training_data.y_train,
            self.EXPECTED_Y_TRAIN_WHEN_NO_WORDS_REMOVED)

    def test_X_train_correctly_encoded_as_indices_when_some_words_removed(self):
        training_data = WordLevelRNNTrainingDataBuilder.build(
            corpus=self.CORPUS,
            vocabulary_size=self.VOCABULARY_SIZE_WHEREBY_SOME_WORDS_REMOVED
        )

        self.assertEqual(training_data.X_train,
            self.EXPECTED_X_TRAIN_WHEN_SOME_WORDS_REMOVED)

    def test_y_train_correctly_encoded_as_indices_when_some_words_removed(self):
        training_data = WordLevelRNNTrainingDataBuilder.build(
            corpus=self.CORPUS,
            vocabulary_size=self.VOCABULARY_SIZE_WHEREBY_SOME_WORDS_REMOVED
        )

        self.assertEqual(training_data.y_train,
            self.EXPECTED_Y_TRAIN_WHEN_SOME_WORDS_REMOVED)
