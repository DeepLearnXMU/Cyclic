import torch
import numpy as np
from torchtext import data
import logging
import random
import argparse

from data.data import NormalField, ParallelDataset
from data.lazy_iterator import BucketIterator, Iterator
import time

from model.dl4nmt import train
from pathlib import Path
import json
import os
from torchtext.vocab import Vectors


def dyn_batch_with_padding(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    return max(len(new.src), len(new.tgt), prev_max_len) * i


def dyn_batch_without_padding(new, i, sofar):
    return sofar + max(len(new.src), len(new.tgt))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer / FastTransformer.')

    # added by lemon
    msg = 'source embedding size, default 620'
    parser.add_argument('--embdim', nargs=1, type=int, help=msg)
    msg = 'source, target hidden size, default 1000'
    parser.add_argument('--hidden', nargs=4, type=int, help=msg)
    msg = 'max epoch'
    parser.add_argument('--maxepoch', type=int, help=msg)
    msg = 'decay'
    parser.add_argument('--decay', type=float, help=msg)
    msg = 'test_corpus'
    parser.add_argument('--test_corpus', type=str, help=msg)
    msg = 'init source vocab'
    parser.add_argument('--init_src_vocab', type=str, help=msg)
    msg = 'patience for early stop'
    parser.add_argument('--patience', type=int, help=msg)
    parser.add_argument('--k', type=int, help=msg)

    # dataset settings
    parser.add_argument('--corpus_prex', type=str)
    parser.add_argument('--lang', type=str, nargs='+', help="the suffix of the corpus, translation language")
    parser.add_argument('--valid', type=str)

    parser.add_argument('--writetrans', type=str, help='write translations for to a file')
    parser.add_argument('--ref', type=str, help='references, word unit')

    parser.add_argument('--vocab', type=str)
    parser.add_argument('--vocab_size', type=int, default=40000)

    parser.add_argument('--load_vocab', action='store_true', help='load a pre-computed vocabulary')
    # parser.add_argument('--load_corpus', action='store_true', default=False, help='load a pre-processed corpus')
    # parser.add_argument('--save_corpus', action='store_true', default=False, help='save a pre-processed corpus')
    parser.add_argument('--max_len', type=int, default=None, help='limit the train set sentences to this many tokens')
    # parser.add_argument('--max_train_data', type=int, default=None,
    #                     help='limit the train set sentences to this many sentences')
    parser.add_argument('--pool', type=int, default=100, help='shuffle batches in the pool')

    # model name
    parser.add_argument('--model', type=str, default='[time]', help='prefix to denote the model, nothing or [time]')

    # network settings
    parser.add_argument('--share_embed', action='store_true', default=False,
                        help='share embeddings and linear out weight')
    parser.add_argument('--share_vocab', action='store_true', default=False,
                        help='share vocabulary between src and target')

    # parser.add_argument('--ffw_block', type=str, default="residual", choices=['residual', 'highway', 'nonresidual'])

    # parser.add_argument('--posi_kv', action='store_true', default=False,
    #                     help='incorporate positional information in key/value')

    # parser.add_argument('--params', type=str, default='user', choices=['user', 'small', 'middle', 'big'],
    #                     help='Defines the dimension size of the parameter')
    # parser.add_argument('--n_layers', type=int, default=5, help='number of layers')
    # parser.add_argument('--n_heads', type=int, default=2, help='number of heads')
    # parser.add_argument('--d_model', type=int, default=278, help='dimention size for hidden states')
    # parser.add_argument('--d_hidden', type=int, default=507, help='dimention size for FFN')

    # parser.add_argument('--initnn', default='standard', help='parameter init')

    # running setting
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test',
                                 'distill'])  # distill : take a trained AR model and decode a training set
    parser.add_argument('--seed', type=int, default=19920206, help='seed for randomness')

    parser.add_argument('--keep_cpts', type=int, default=1, help='save n checkpoints, when 1 save best model only')

    # training
    parser.add_argument('--eval_every', type=int, default=100, help='validate every * step')
    parser.add_argument('--save_every', type=int, default=-1, help='save model every * step (5000)')

    parser.add_argument('--batch_size', type=int, default=2048, help='# of tokens processed per batch')
    parser.add_argument('--delay', type=int, default=1, help='gradiant accumulation for delayed update for large batch')

    parser.add_argument('--optimizer', type=str, default='Adadelta')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--warmup', type=int, default=4000, help='maximum steps to linearly anneal the learning rate')

    parser.add_argument('--maximum_steps', type=int, default=5000000, help='maximum steps you take to train a model')
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--input_drop_ratio', type=float, default=0.1, help='dropout ratio only for inputs')

    parser.add_argument('--grad_clip', type=float, default=-1.0, help='gradient clipping')
    parser.add_argument('--smoothing', type=float, default=0.0, help='label smoothing')

    # decoding
    parser.add_argument('--length_ratio', type=float, default=2, help='maximum lengths of decoding')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='beam-size used in Beamsearch, default using greedy decoding')
    parser.add_argument('--alpha', type=float, default=0.6, help='length normalization weights')

    parser.add_argument('--test', type=str, default=None, help='test src file')

    # model saving/reloading, output translations
    parser.add_argument('--load_from', nargs='+', default=None, help='load from 1.modelname, 2.lastnumber, 3.number')

    parser.add_argument('--resume', action='store_true',
                        help='when resume, need other things besides parameters')
    # save path
    parser.add_argument('--main_path', type=str, default="./")
    parser.add_argument('--model_path', type=str, default="models")
    parser.add_argument('--decoding_path', type=str, default="decoding")

    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def curtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def override(args, load_dict, except_name):
    for k in args.__dict__:
        if k not in except_name:
            args.__dict__[k] = load_dict[k]


'''
You can call `torch.load(.., map_location='cpu')`
and then :meth:`load_state_dict` to avoid GPU RAM surge when loading a model checkpoint
'''

if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'train':
        if args.load_from is not None and len(args.load_from) == 1:
            load_from = args.load_from[0]
            print('{} load the checkpoint from {} for initilize or resume'.
                  format(curtime(), load_from))
            checkpoint = torch.load(load_from, map_location='cpu')
        else:
            checkpoint = None

        # if not resume(initilize), only need model parameters
        if args.resume:
            print('update args from checkpoint')
            load_dict = checkpoint['args'].__dict__
            except_name = ['mode', 'resume', 'maximum_steps']
            override(args, load_dict, tuple(except_name))

        main_path = Path(args.main_path)
        model_path = main_path / args.model_path
        decoding_path = main_path / args.decoding_path

        for path in [model_path, decoding_path]:
            path.mkdir(parents=True, exist_ok=True)

        args.model_path = str(model_path)
        args.decoding_path = str(decoding_path)

        if args.model == '[time]':
            args.model = time.strftime("%m.%d_%H.%M.", time.gmtime())

        # setup random seeds
        set_seeds(args.seed)

        DataField = NormalField
        SRC = DataField(batch_first=True)
        TRG = DataField(unk_token=None, init_token="<START>", pad_token=None, eos_token="<STOP>", batch_first=True)
        TAG = DataField(batch_first=True, use_vocab=False, sequential=False)

        train_data = ParallelDataset(path=args.corpus_prex, exts=args.lang, fields=(SRC, TRG, TAG),
                                     max_len=args.max_len)

        dev_data = ParallelDataset(path=args.valid, exts=args.lang, fields=(SRC, TRG, TAG))

        test_data = ParallelDataset(path=args.test_corpus, exts=args.lang, fields=(SRC, TRG, TAG))

        # if args.init_src_vocab:
        #     # cache = "/home/lemon/data/Ch-simile"
        #     cache = Path(args.vocab)
        #     src_vocab_path = cache / '{}'.format(args.init_src_vocab)
        #     if not os.path.exists(cache):
        #         os.mkdir(cache)
        #     vectors = Vectors(name=src_vocab_path, cache=cache)
        #     SRC.build_vocab(train_data, vectors=vectors)

        # build vocabularies for translation dataset
        vocab_path = Path(args.vocab)
        tgt_vocab_path = vocab_path / '{}.pt'.format(args.lang[1])
        # tag_vocab_path = vocab_path / '{}.pt'.format(args.lang[2])

        if args.load_vocab and tgt_vocab_path.exists():
            TRG.vocab = torch.load(str(tgt_vocab_path))
            # TAG.vocab = torch.load(str(tag_vocab_path))
            print('vocab {} loaded'.format(str(vocab_path)))
        else:
            assert (not train_data is None)
            TRG.build_vocab(train_data, max_size=args.vocab_size)
            # TAG.build_vocab(train_data, max_size=args.vocab_size)

            print('save the vocabulary')
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(TRG.vocab, str(tgt_vocab_path))
            # torch.save(TAG.vocab, str(tag_vocab_path))

        args.__dict__.update({'tgt_vocab': len(TRG.vocab)})

        train_flag = True
        train_real = BucketIterator(train_data, args.batch_size, device='cuda',
                                    train=train_flag, repeat=False,
                                    shuffle=train_flag, poolnum=args.pool,sort_within_batch=True)

        devbatch = 20
        dev_real = Iterator(dev_data, devbatch, device='cuda', batch_size_fn=None,
                            train=False, repeat=False, shuffle=False, sort=False)

        testbatch = 20
        test_real = Iterator(test_data, testbatch, device='cuda', batch_size_fn=None,
                            train=False, repeat=False, shuffle=False, sort=False)

        args_str = json.dumps(args.__dict__, indent=4, sort_keys=True)
        print(args_str)

        print('{} Starti training'.format(curtime()))
        # args, train_iter, dev, test, src_field, tgt_field, checkpoint
        train(args, train_real, dev_real, test_real, SRC, TRG, TAG, checkpoint)
