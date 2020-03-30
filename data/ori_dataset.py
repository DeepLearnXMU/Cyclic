import torch
import _pickle as pickle
# import revtok
import os

from torchtext import data, datasets
from contextlib import ExitStack
import fractions

try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction


# load the dataset + reversible tokenization
class NormalField(data.Field):
    def reverse(self, batch, unbpe=True, returen_token=False):
        if not self.batch_first:
            batch.t_()

        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        if unbpe:
            batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
        else:
            batch = [" ".join(filter(filter_special, ex)) for ex in batch]

        if returen_token:
            batch = [ex.split() for ex in batch]
        return batch


class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path, exts, fields, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.
        Arguments:
            root: Root dataset storage directory. Default is '.data'.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class ParallelDataset(datasets.TranslationDataset):
    """ Define a N-parallel dataset: supports abitriry numbers of input streams"""

    def __init__(self, path=None, exts=None, fields=None,
                 load_dataset=False, save_dataset=False, prefix='', examples=None, **kwargs):

        if examples is None:
            assert len(exts) == len(fields), 'N parallel dataset must match'
            self.N = len(fields)

            if not isinstance(fields[0], (tuple, list)):
                newfields = [('src', fields[0]), ('trg', fields[1])]
                for i in range(len(exts) - 2):
                    newfields.append(('extra_{}'.format(i), fields[2 + i]))
                self.fields = newfields

            paths = tuple(os.path.expanduser(path + '.' + x) for x in exts)
            if load_dataset and (os.path.exists(path + '.processed.{}.pt'.format(prefix))):
                examples = torch.load(path + '.processed.{}.pt'.format(prefix))
            else:
                examples = []
                with ExitStack() as stack:
                    files = [stack.enter_context(open(fname, 'r', errors='ignore')) for fname in paths]
                    for i, lines in enumerate(zip(*files)):
                        lines = [line.strip() for line in lines]
                        if not any(line == '' for line in lines):
                            examples.append(data.Example.fromlist(lines, fields))
                if load_dataset:
                    torch.save(examples, path + '.processed.{}.pt'.format(prefix))

        super(datasets.TranslationDataset, self).__init__(examples, fields, **kwargs)
