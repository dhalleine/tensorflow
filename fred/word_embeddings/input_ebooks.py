import collections
import numpy as np
import random
import os
import re

EBOOKS_PATH = "/home/fred/ebooks/"
NB_FILES_TO_INJEST = 1
VOCABULARY_SIZE = 1000
UNKNOWN_WORD = "_UNKNOWN_"

def read_corpus(path, files_to_read):
    data = []
    counter = collections.Counter()
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            book_data, book_counter = read_ebook(path + filename)
            data.extend(book_data)
            counter = counter + book_counter
            files_to_read -= 1
            if files_to_read <= 0:
                break
    return data, counter

def extract_word(line):
    return [w for w in re.split("[^a-z]+", line.strip().lower()) if w]

def read_ebook(filename):
    print "Reading %s" % filename
    data = []
    counter = collections.Counter()
    LINES_DEBUG = 500
    with open(filename) as ebook_file:
        # Split the file by punctuation and make an array of words
        for words in [extract_word(line) for line in re.split("[\\.\\?\\!\\:]", ebook_file.read())]:
            if words:
                data.append(words)
                counter = counter + collections.Counter(words)
            LINES_DEBUG -= 1
            if LINES_DEBUG < 0:
                break;
    print "%d sentences." % len(data)
    return data, counter

def build_dataset(words_list, counter, vocabulary_size):
    print "Counting words frequencies"
    # TODO: move this counting words while reading the ebooks
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
    def __init__(self, data, count, dictionary, reverse_dictionary):
        self.data = data
        self.count = count
        self.vocabulary_size = len(dictionary)
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self._data_index = 0

    def get_batch(self, batch_size, num_skips, skip_window):
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

def read_data_sets():
    words, counter = read_corpus(EBOOKS_PATH, NB_FILES_TO_INJEST)
    data, count, dictionary, reverse_dictionary = build_dataset(words, counter, VOCABULARY_SIZE)
    return DataSets(data, count, dictionary, reverse_dictionary)


