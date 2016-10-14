from collections import namedtuple
import itertools
import nltk


class WordLevelRNNTrainingDataBuilder:

    UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
    NUMBER_OF_WORDS_TO_ADD_IN_MANUALLY = len(['UNKNOWN_TOKEN', 'SENTENCE_START', 'SENTENCE_END'])
    WORD_COUNT_ITEM = namedtuple('WordCountItem', ['word', 'count'])

    @classmethod
    def build(cls, corpus, vocabulary_size):
        tokenized_corpus = cls._tokenize_corpus_into_list_of_tokenized_sentences(corpus)
        tokenized_corpus = cls._remove_uncommon_words(tokenized_corpus, vocabulary_size)
        tokenized_corpus = cls._append_sentence_start_and_end_tokens(tokenized_corpus)
        return _RNNTrainingData(tokenized_corpus=tokenized_corpus)

    @classmethod
    def _tokenize_corpus_into_list_of_tokenized_sentences(cls, corpus):
        tokenized_corpus = nltk.sent_tokenize(corpus)
        tokenized_corpus = [cls._clean_sentence(sentence) for sentence in tokenized_corpus]
        return [nltk.word_tokenize(sentence) for sentence in tokenized_corpus]

    @classmethod
    def _remove_uncommon_words(cls, tokenized_corpus, vocabulary_size):
        word_count = nltk.FreqDist( itertools.chain(*tokenized_corpus) )
        word_count = [cls.WORD_COUNT_ITEM(word=word, count=count) for word, count in word_count.items()]
        word_count = sorted(word_count, key=lambda item: (item.count, item.word), reverse=True)
        most_common_words = [word_count_item.word for word_count_item in word_count[:vocabulary_size - \
            cls.NUMBER_OF_WORDS_TO_ADD_IN_MANUALLY + 1]]

        tokenized_corpus = [
            [word if word in most_common_words else cls.UNKNOWN_TOKEN for word in sentence]\
            for sentence in tokenized_corpus
        ]
        return tokenized_corpus

    @staticmethod
    def _clean_sentence(sentence):
        sentence = ' '.join(sentence.split())
        return sentence.lower()

    @staticmethod
    def _append_sentence_start_and_end_tokens(tokenized_corpus):
        return [['SENTENCE_START'] + sentence + ['SENTENCE_END'] for sentence in tokenized_corpus]


class _RNNTrainingData:

    def __init__(self, tokenized_corpus):
        self.training_data_as_tokens = tokenized_corpus
        self.token_to_index_lookup = self._compose_token_to_index_lookup(tokenized_corpus)
        self.index_to_token_lookup = self._compose_index_to_token_lookup()
        self.training_data_as_indices = self._indexify_tokenized_corpus()
        self.X_train = self._compose_X_train()
        self.y_train = self._compose_y_train()

    def _indexify_tokenized_corpus(self):
        return [[self.token_to_index_lookup[token] for token in sentence] \
                for sentence in self.training_data_as_tokens]

    def _compose_X_train(self):
        return [training_instance[:-1] for training_instance in self.training_data_as_indices]

    def _compose_y_train(self):
        return [training_instance[1:] for training_instance in self.training_data_as_indices]

    def _compose_index_to_token_lookup(self):
        return {index: token for token, index in self.token_to_index_lookup.items()}

    @staticmethod
    def _compose_token_to_index_lookup(tokenized_corpus):
        unique_tokens = set( itertools.chain(*tokenized_corpus) )
        unique_tokens = list(unique_tokens)
        unique_tokens.sort()
        return {token: index for index, token in enumerate(unique_tokens)}
