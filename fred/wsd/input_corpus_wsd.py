import collections
import numpy as np
import random
import os
import re

INPUT_PATH_PATTERN = "/home/fred/corpus/wsd/%s.txt"
STOP_WORDS_PATH = "data/stop_words.txt"
VOCABULARY_SIZE = 1000
UNKNOWN_WORD = "_UNKNOWN_"

def read_corpus(word_to_disambiguate, corpus_filename):
    print "Reading %s" % corpus_filename
    data = []
    counter = collections.Counter()
    LINES_DEBUG = 500
    with open(corpus_filename) as corpus_file:
        # Split the file by punctuation and make an array of words
        for words in [extract_word(line) for line in re.split("[\\.\\?\\!\\:]", corpus_file.read())]:
            if words:
                data.append(words)
                counter = counter + collections.Counter(words)
            LINES_DEBUG -= 1
            if LINES_DEBUG < 0:
                break;
    print "%d sentences." % len(data)
    return data, counter

def extract_word(line):
    return [w for w in re.split("[^a-z]+", line.strip().lower()) if w]

def build_dataset(word_to_disambiguate, words_list, counter, vocabulary_size):
    # Remove the stop words
    for stop_word in open(STOP_WORDS_PATH).read().split():
        del counter[stop_word]
    # Remove the word to disambiguate
    del counter[word_to_disambiguate]
    count = [[UNKNOWN_WORD, -1]]
    count.extend(counter.most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for words in words_list:
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary[UNKNOWN_WORD]
                unk_count = unk_count + 1
            data.append(index)
        # Append with unknow words to make the border between two sentences
        for i in range(2):
            data.append(0)
            unk_count = unk_count + 1
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    del words_list
    print "Most common words: %s" % count[:20]
    print "Sample data: %s" % data[:5]
    return data, count, dictionary, reverse_dictionary

class DataSets(object):
    def __init__(self, word_to_disambiguate, data, count, dictionary, reverse_dictionary):
        self.word_to_disambiguate = word_to_disambiguate
        self.data = data
        self.count = count
        self.vocabulary_size = len(dictionary)
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self._data_index = 0

    def debug(self):
        for i, w in self.dictionary.iteritems():
            print "Word %d = %s" % (i, w)
        for i, w in self.reverse_dictionary.iteritems():
            print "Word %d = %s" % (i, w)

    def get_batch_1(self, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self._data_index])
            self._data_index = (self._data_index + 1) % len(self.data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self._data_index])
            self._data_index = (self._data_index + 1) % len(self.data)
        return batch, labels

def read_data_sets(word_to_disambiguate):
    words, counter = read_corpus(word_to_disambiguate, INPUT_PATH_PATTERN % word_to_disambiguate)
    data, count, dictionary, reverse_dictionary = build_dataset(word_to_disambiguate, words, counter, VOCABULARY_SIZE)
    return DataSets(word_to_disambiguate, data, count, dictionary, reverse_dictionary)


