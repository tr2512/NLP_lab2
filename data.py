from collections import Counter, OrderedDict, defaultdict, namedtuple
from nltk import Tree
import torch
import re
import random
import numpy as np

def filereader(path):
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\","")

def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.sub(r"\([0-9] |\)", "", s).split()

def transitions_from_treestring(s):
    s = re.sub("\([0-5] ([^)]+)\)", "0", s)
    s = re.sub("\)", " )", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\)", "1", s)
    return list(map(int, s.split()))

def examplereader(path, lower=False):
    """Returns all examples in a file one by one."""
    Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])

    for line in filereader(path):
        line = line.lower() if lower else line
        tokens = tokens_from_treestring(line)
        tree = Tree.fromstring(line)  # use NLTK's Tree
        label = int(line[1])
        trans = transitions_from_treestring(line)
        yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)



class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))
    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""
    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        '''
        min_freq: minimum number of occurrences for a word to be included
                in the vocabulary
        '''
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)


def build_vocab(train_data):
    v = Vocabulary()
    for data_set in (train_data,):
        for ex in data_set:
            for token in ex.tokens:
                v.count_token(token)

    v.build()
    print("Vocabulary size:", len(v.w2i))
    return v

def prepare_example(example, vocab, device):
    """
    Map tokens to their IDs for a single example
    """

    # vocab returns 0 if the word is not there (i2w[0] = <unk>)
    x = [vocab.w2i.get(t, 0) for t in example.tokens]

    x = torch.LongTensor([x])
    x = x.to(device)

    y = torch.LongTensor([example.label])
    y = y.to(device)

    return x, y

def get_examples(data, shuffle=True, **kwargs):
    """Shuffle data set and return 1 example at a time (until nothing left)"""
    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch
    for example in data:
        yield example

def prepare_minibatch(mb, vocab, device):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]

    x = torch.LongTensor(x)
    x = x.to(device)

    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)

    return x, y

def get_minibatch(data, batch_size=25, shuffle=True):
    """Return minibatches, optional shuffling"""

    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch

def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))

def prepare_treelstm_minibatch(mb, vocab, device):
    """
    Returns sentences reversed (last word first)
    Returns transitions together with the sentences.
    """
    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    # NOTE: reversed sequence!
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb]

    x = torch.LongTensor(x)
    x = x.to(device)

    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)

    maxlen_t = max([len(ex.transitions) for ex in mb])
    transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
    transitions = np.array(transitions)
    transitions = transitions.T  # time-major

    return (x, transitions), y
