import math
# import ipdb
from collections import OrderedDict, Counter
from itertools import chain

import torch
import os

from torchtext import data, datasets
from torchtext.data import Example
from contextlib import ExitStack

# load the dataset + reversible tokenization
class NormalField(data.Field):

    def _getattr(self, dataset, attr):
            for x in dataset:
                yield getattr(x, attr)

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            sources += [self._getattr(arg, name) for name, field in
                            arg.fields if field is self]
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def reverse(self, batch, unbpe=True, returen_token=False):
        if not self.batch_first:
            batch.t_()

        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch] # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch] # trim past frst eos
        
        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        if unbpe:
            batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
        else:
            batch = [" ".join(filter(filter_special, ex)) for ex in batch]

        if returen_token:
            batch = [ex.split() for ex in batch]
        return batch


class ParallelDataset(object):
    """ Define a N-parallel dataset: supports abitriry numbers of input streams"""

    def __init__(self, path=None, exts=None, fields=None, max_len=None):
        assert len(exts) == len(fields), 'N parallel dataset must match'
        self.N = len(fields)

        if not isinstance(fields[0], (tuple, list)):
            newfields = [('src', fields[0]), ('tgt', fields[1]), ('tag', fields[2])]
            for i in range(len(exts) - 3):
                newfields.append(('extra_{}'.format(i), fields[3 + i]))
            self.fields = newfields
        self.paths = tuple(os.path.expanduser(path + '.' + x) for x in exts)
        self.max_len = max_len

    def __iter__(self):
        with ExitStack() as stack:
            files = [stack.enter_context(open(fname, 'r', errors='ignore')) for fname in self.paths]
            for i, lines in enumerate(zip(*files)):
                lines = [line.strip() for line in lines]
                if not any(line == '' for line in lines):
                    example = Example.fromlist(lines, self.fields)
                    if self.max_len is None:
                        yield example
                    elif len(example.src) <= self.max_len and len(example.tgt) <= self.max_len:
                        yield example
