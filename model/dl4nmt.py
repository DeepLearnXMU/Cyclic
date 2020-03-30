# lemon
# lemon@stu.xmu.edu.cn

import re
import time
import torch
import subprocess
import itertools
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence as unpack, \
    pack_padded_sequence as pack
from search.search import greedy_search,beam_search
from sklearn.metrics import precision_score, recall_score, f1_score
from model.paireval import evaluate

#from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_transformers import BertTokenizer, BertModel
from pytorch_transformers import AdamW, WarmupLinearSchedule

def print_params(model):
    print('total_params', sum([np.prod(list(p.size())) for p in model.parameters()]))
    print('enc parameters:', sum([np.prod(list(p.size())) for p in model.encoder.parameters()]))

def prepare_src(data, padid):
    assert data.dim() == 2
    mask = (data != padid)

    return data, mask

def prepare_tgt(data):
    assert data.dim() == 2
    data = data[:,1:-1]
    return data

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, dim=-1)
    return idx

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    _, B, tgt_size = vec.size()
    max_score = vec[0, range(B), argmax(vec)]

    max_score_broadcast = max_score.view(B, -1).expand(-1, tgt_size)

    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast),dim=-1))

class Self_Neural_Attention(nn.Module):
    "e_i = v_a^T tanh(map_key_i); alpha = softmax({e_i})"

    def __init__(self, dim_value, dim_atten):
        super(Self_Neural_Attention, self).__init__()
        self.map_values = nn.Linear(dim_value, dim_atten, bias=False)
        self.pre_alpha = nn.Linear(dim_atten, 1, bias=False)

    def __call__(self, values, mask):
        # B, T, H
        time_dim = 1
        mapped_values = self.map_values(values) # B, T, H

        act = torch.tanh(mapped_values)
        e = self.pre_alpha(act).squeeze(-1) # B, T
        e.masked_fill_(mask == 0, -float('inf')) # B, T
        alpha = F.softmax(e, dim=time_dim) # T, B
        output = torch.bmm(alpha.unsqueeze(1), values).squeeze(1)
        return output, alpha

class LSTMEncoder(nn.Module):
    "Bi-derectional GRU Encoder"

    def __init__(self, sedim, shdim):
        super(LSTMEncoder, self).__init__()
        self.sedim = sedim
        self.shdim = shdim
        self.bi_cell = nn.LSTM(sedim, shdim, batch_first=True, bidirectional=True)

    def forward(self, x):

        states, _ = self.bi_cell(x)

        return states

class Extractor(nn.Module):
    def __init__(self, shdim, thdim, tgt_vocab_size, tgt_field):
        super(Extractor, self).__init__()

        # padid
        self.tgt_field = tgt_field
        self.tag_to_ix = tgt_field.vocab.stoi
        # tags
        self.START_TAG = tgt_field.init_token
        self.STOP_TAG = tgt_field.eos_token

        # active layer
        activation = nn.Linear(shdim, thdim)

        # proj
        hidden2tag = nn.Linear(thdim, tgt_vocab_size, bias=False)

        # Matrix of transition parameters.
        self.transitions = nn.Parameter(
            torch.randn(tgt_vocab_size, tgt_vocab_size))

        self.tgt_vocab_size = tgt_vocab_size
        self.activation = activation
        self.hidden2tag = hidden2tag

    def constrain_transition(self):
        TS = self.tgt_field.vocab.stoi['ts']
        VS = self.tgt_field.vocab.stoi['vs']
        VB = self.tgt_field.vocab.stoi['vb']
        VE = self.tgt_field.vocab.stoi['ve']
        TB = self.tgt_field.vocab.stoi['tb']
        TE = self.tgt_field.vocab.stoi['te']
        VM = self.tgt_field.vocab.stoi['vm']
        TM = self.tgt_field.vocab.stoi['tm']
        O = self.tgt_field.vocab.stoi['O']
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.tag_to_ix[self.START_TAG], :] = -1000
        self.transitions.data[:, self.tag_to_ix[self.STOP_TAG]] = -1000
        self.transitions.data[[TB, TS, VM, VS, VB, VE], TB] = -1000
        self.transitions.data[[TB, TS, VB, VM, VE, VS], TM] = -1000
        self.transitions.data[[TE, TB, TM, TS], TE] = -1000
        self.transitions.data[[TS, TB, TM, TE, VM, VE], TS] = -1000
        self.transitions.data[[TS, TB, TM, TE, VS], VB] = -1000
        self.transitions.data[[TB, TM, TE, TS, VB, VS], VM] = -1000
        self.transitions.data[[VE, VB, VM, VS], VE] = -1000
        self.transitions.data[[TM, TE, VS, VB, VM, VE], VS] = -1000
        self.transitions.data[[TM, TE, VM, VE], O] = -1000

    def _forward_alg(self, feats, masks):
        '''
        :param feats: L, B
        :return:
        '''
        L, B, tagset_size = feats.size()
        # Do the forward algorithm to compute the partition function
        init_alphas = feats.new_zeros((1, B, tagset_size)).fill_(-10000.)

        # START_TAG has all of the score.
        init_alphas[0, range(B), self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for idx, feat in enumerate(feats):
            mask = masks[idx].view(B, -1).expand(-1, tagset_size).bool()
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(tagset_size):
                emit_score = feat[range(B), next_tag].view(B,-1).expand(-1, tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var))

            forward_var = forward_var.masked_fill(mask, 0.) + \
                          torch.cat(alphas_t).transpose(1, 0).view(1, B, -1).masked_fill(~mask, 0.)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        alpha = log_sum_exp(terminal_var.view(1, B, -1))

        return alpha[0]

    def _score_sentence(self, feats, tags, masks):
        '''
        :param feats: L, B, tag_size
        :param tags: L, B
        :return:
        '''
        L, B, tagset_size = feats.size()
        # Gives the score of a provided tag sequence
        score = feats.new_zeros(B)
        start_tensor = torch.tensor([self.tag_to_ix[self.START_TAG] for i in range(B)],dtype=torch.long).cuda()
        tags = torch.cat([start_tensor.view(B,1), tags], dim=-1)

        for i, feat in enumerate(feats):
            mask = masks[i].bool()
            score = score + \
                    (self.transitions[tags[range(B), i + 1], tags[range(B), i]] + \
                     feat[range(B), tags[range(B), i + 1]]).masked_fill(~mask, 0)

        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[range(B), masks.sum(0)]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        L, tagset_size = feats.size()
        # Initialize the viterbi variables in log space
        init_vvars = feats.new_zeros((1, tagset_size)).fill_(-10000.)
        # init_vvars = torch.full((1, tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var).item()
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var).item()
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    # def forward(self, src_annots, mask, tgt):
    #     B, L = mask.size()
    #     mask = mask.transpose(1,0) # L, B
    #     feature = torch.tanh(self.activation(src_annots))
    #     feats = self.hidden2tag(feature)
    #     feats = feats.transpose(1, 0) # transfer to (L,B,tag_size) from (B,L, tag_size)
    #
    #     forward_score = self._forward_alg(feats, mask)
    #     gold_score = self._score_sentence(feats, tgt, mask)
    #     loss = (forward_score - gold_score).sum() / B
    #     return loss
    def forward(self, efeats, mask, tgt):
        B, L = mask.size()
        mask = mask.transpose(1, 0)  # L, B
        efeats = efeats.transpose(1, 0)  # transfer to (L,B,tag_size) from (B,L, tag_size)

        forward_score = self._forward_alg(efeats, mask)
        gold_score = self._score_sentence(efeats, tgt, mask)
        loss = (forward_score - gold_score).sum() / B
        return loss
    def get_feature(self, src_annots):
        feature = torch.tanh(self.activation(src_annots))
        feats = self.hidden2tag(feature)
        return feats

    # def decoding(self, src_annots, masks):
    #     """
    #     :param src_annots:  [batch, length, shdim]
    #     :param masks: [batch, length]
    #     :return:
    #     """
    #     B, L = masks.size()
    #     # encoding the source
    #     feature = torch.tanh(self.activation(src_annots))
    #     feats = self.hidden2tag(feature)
    #
    #     scores = []
    #     tag_seqs = []
    #     for i in range(B):
    #         feat = feats[i, :masks.sum(1)[i]]
    #         score, tag_seq = self._viterbi_decode(feat)
    #         scores.append(score.item())
    #         tag_seqs.append(tag_seq)
    #     return scores, tag_seqs

    def decoding(self, feats, masks):
        """
        :param src_annots:  [batch, length, shdim]
        :param masks: [batch, length]
        :return:
        """
        B, L = masks.size()
        # # encoding the source
        # feature = torch.tanh(self.activation(src_annots))
        # feats = self.hidden2tag(feature)

        scores = []
        tag_seqs = []
        for i in range(B):
            feat = feats[i, :masks.sum(1)[i]]
            score, tag_seq = self._viterbi_decode(feat)
            scores.append(score.item())
            tag_seqs.append(tag_seq)
        return scores, tag_seqs

class Classify(nn.Module):
    def __init__(self, shdim, thdim, ahdim, tag_vocab_size):
        super(Classify, self).__init__()

        # Self-attention
        attention = Self_Neural_Attention(shdim, thdim)

        # active layer
        activation = nn.Linear(shdim, ahdim)

        # proj
        proj = nn.Linear(ahdim, tag_vocab_size, bias=False)

        self.attention = attention
        self.activation = activation
        self.proj = proj
        self.nll_loss = nn.NLLLoss()

    def loss(self, probs, truth):
        loss = self.nll_loss(probs.contiguous().view(-1, probs.size(-1)),
                             truth.contiguous().view(-1))

        return loss

    def pred_tag(self, context):
        # context, _ = self.attention(src_annots, src_mask)
        feature = torch.tanh(self.activation(context))
        probs = self.proj(feature)
        idx = torch.argmax(probs, dim=-1)
        return idx

    def get_feature(self, src_annots, src_mask):
        # attention
        context, alpha = self.attention(src_annots, src_mask)  # [batch, shdim]
        # _, alpha = self.attention(src_annots, src_mask)  # [batch, shdim]
        return context, alpha

    def forward(self, context, tag):
        feature = torch.tanh(self.activation(context))

        feats = self.proj(feature) # [batch, thdim]

        log_probs = F.log_softmax(feats, dim=-1)

        loss = self.loss(log_probs, tag)

        return loss

class Classify_Extractor(nn.Module):
    def __init__(self, args, tgt_field):
        super(Classify_Extractor, self).__init__()

        sedim = args.embdim[0]

        # hidden layer dim
        shdim, cthdim, cahdim, ethdim = args.hidden

        # vocab size
        # src_vocab_size = args.src_vocab
        tgt_vocab_size = args.tgt_vocab
        # tag_vocab_size = args.tag_vocab

        self.bert = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True, output_attentions=False)
        # self.bert.cuda()
        # bert_dim = 768

        # self.bert2enc = nn.Linear(bert_dim, sedim)
        # self.bert.eval()
        sedim = 768

        # padid
        # self.srcpadid = srcpadid
        self.tgt_field = tgt_field
        self.tag_to_ix = tgt_field.vocab.stoi
        # tags
        self.START_TAG = tgt_field.init_token
        self.STOP_TAG = tgt_field.eos_token

        # dropout
        self.drop = nn.Dropout(args.drop_ratio)
        self.tgt_vocab_size = tgt_vocab_size

        # Bi-directional LSTM encoder
        encoder = LSTMEncoder(sedim, shdim)

        # Classifier
        classifier = Classify(shdim * 2 +tgt_vocab_size, cthdim, cahdim, 2)

        # Extractor
        extractor = Extractor(shdim * 2 + 1, ethdim, tgt_vocab_size, tgt_field)
        extractor.constrain_transition()

        # self.src_emb = src_emb
        self.encoder = encoder
        self.classifier = classifier
        self.extractor = extractor
        self.k = args.k

    def bert_emb(self, src, mask):
        # with torch.no_grad():
        if True:
            output = self.bert(src, token_type_ids=None, attention_mask=mask)[2]
        #     encoded_layers, _ = self.bert(src, token_type_ids=None, attention_mask=mask)
            # use [CLS] for sentence embedding
            #h = encoded_layers[-1][:, 0]
            # B T H
            last_h = output[-1]
        return last_h

    def bert_encoding(self, src, src_mask):
        # batch_first = True
        bert_embed = self.bert_emb(src, src_mask)
        bert_embed = self.drop(bert_embed)

        length = src_mask.sum(-1)

        sorted_len, ix = torch.sort(length, descending=True)
        sorted_bert_embed = bert_embed[ix]

        packed_sorted_bert_embed = nn.utils.rnn.pack_padded_sequence(sorted_bert_embed, sorted_len, True)
        packed_annotations = self.encoder(packed_sorted_bert_embed)
        annotations, _ = nn.utils.rnn.pad_packed_sequence(packed_annotations, True)

        _, recovered_ix = torch.sort(ix, descending=False)
        annotations = annotations[recovered_ix]

        annotations = self.drop(annotations)
        return annotations

    def encoding(self, src):
        """
        :param src: (B, T)
        :return: annotations (B, T, 2*shdim);
        """
        batch_first = True

        seq_lens = (src != self.srcpadid).sum(1) # B, 1

        sorted_seq_lens, indices = torch.sort(seq_lens, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)

        if batch_first:
            src = src[indices]
        else:
            src = src[:, indices]

        packed_inputs = pack(src, sorted_seq_lens, batch_first=batch_first)

        idx = packed_inputs.data
        src_embedding = self.src_emb(idx)
        src_embedding = self.drop(src_embedding)

        src_embedding = PackedSequence(src_embedding, packed_inputs.batch_sizes)

        annotations = self.encoder(src_embedding)
        unpacked_inputs, _ = unpack(annotations, batch_first=batch_first)

        if batch_first:
            annotations = unpacked_inputs[desorted_indices]
        else:
            annotations = unpacked_inputs[:, desorted_indices]

        annotations = self.drop(annotations)

        return annotations

    def simile_classify(self, src, src_mask):
        annotations = self.bert_encoding(src, src_mask)
        K = self.k
        efeats = torch.zeros([annotations.shape[0], annotations.shape[1], self.tgt_vocab_size]).cuda()
        # cfeats = torch.zeros([annotations.shape[0],annotations.shape[1],self.tgt_vocab_size])
        kcontext = None
        for i in range(K):
            cla_annot = torch.cat([annotations, efeats], dim=-1)
            context, alpha = self.classifier.get_feature(cla_annot, src_mask)
            cfeats = alpha.unsqueeze(-1)
            ext_annot = torch.cat([annotations, cfeats], dim=-1)
            efeats = self.extractor.get_feature(ext_annot)
            if i == 0:
                kcontext = context
            else:
                kcontext += context
        tags = self.classifier.pred_tag(kcontext)

        return tags

    def component_extraction(self, src, src_mask):
        annotations = self.bert_encoding(src, src_mask)
        efeats = torch.zeros([annotations.shape[0], annotations.shape[1], self.tgt_vocab_size]).cuda()
        # cfeats = torch.zeros([annotations.shape[0],annotations.shape[1],self.tgt_vocab_size])
        kefeats = None
        K = self.k
        for i in range(K):
            cla_annot = torch.cat([annotations, efeats], dim=-1)
            context, alpha = self.classifier.get_feature(cla_annot, src_mask)
            cfeats = alpha.unsqueeze(-1)
            ext_annot = torch.cat([annotations, cfeats], dim=-1)
            efeats = self.extractor.get_feature(ext_annot)
            if i == 0:
                kefeats = efeats
            else:
                kefeats += efeats
        scores, tag_seqs = self.extractor.decoding(kefeats, src_mask)

        return scores, tag_seqs

    def forward(self, src, src_mask, tgt, tag):
        """
        :param src: [batch, length]
        :param src_mask: [batch, length]
        :param tgt: [batch, length]
        :param tag: [batch,tag]
        :return:
        """
        # encoding the source
        annotations = self.bert_encoding(src, src_mask)
        K = self.k
        efeats = torch.zeros([annotations.shape[0],annotations.shape[1],self.tgt_vocab_size]).cuda()
        # cfeats = torch.zeros([annotations.shape[0],annotations.shape[1],self.tgt_vocab_size])
        kcontext = None
        kefeats = None
        for i in range(K):
            cla_annot = torch.cat([annotations, efeats], dim=-1)
            context, alpha = self.classifier.get_feature(cla_annot, src_mask)
            cfeats = alpha.unsqueeze(-1)
            ext_annot = torch.cat([annotations, cfeats], dim=-1)
            efeats = self.extractor.get_feature(ext_annot)
            if i == 0:
                kcontext = context
                kefeats = efeats
            else:
                kcontext += context
                kefeats += efeats

        cla_loss = self.classifier(kcontext, tag)

        ext_loss = self.extractor(kefeats, src_mask, tgt)

        loss = 0.2 * cla_loss + 0.8 * ext_loss

        return loss

def train(args, train_iter, dev, test, src_field, tgt_field, tag_field, checkpoint):
    # srcpadid = src_field.vocab.stoi['<pad>']
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    model = Classify_Extractor(args,tgt_field)

    if torch.cuda.is_available():
        model.cuda()

    print_params(model)

    decay = args.decay

    if args.optimizer == 'bert':
        weight_decay = 0.0
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        opt = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        totalnum = 0
        for i in train_iter:
            totalnum += 1
        #print(args.lr)
        #print(args.maximum_steps)
        #exit()
        t_total = totalnum // decay * args.maximum_steps
        scheduler = WarmupLinearSchedule(opt, warmup_steps=0, t_total=t_total)
    else:
        opt = torch.optim.Adadelta(model.parameters(), lr=args.lr)

    best_e = 0.0
    best_c = 0.0
    best_epoch_for_c = 0
    best_epoch_for_e = 0
    offset = 0.0
    pre_epoch = 0
    patience_c = 0
    patience_e = 0

    if checkpoint is not None:
        print('model.load_state_dict(checkpoint[model])')
        model.load_state_dict(checkpoint['model'])
        if args.resume:
            opt.load_state_dict(checkpoint['optim'])

            best_f = checkpoint['f']
            offset = checkpoint['iters']
            pre_epoch = checkpoint['epoch']

            print('*************************************')
            print('resume from {} epoch {} iters and best_f {}'.format(pre_epoch, offset, best_f))
            print('*************************************')

    print("**************start training****************")
    start = time.time()

    for epoch in range(args.maxepoch):
        train_iter.init_epoch()
        epoch += pre_epoch

        for iters, train_batch in enumerate(train_iter):
            iters += offset
            model.train()
            # model.zero_grad()
            # model.constrain_transition()
            t1 = time.time()
            batch_src = train_batch.src
            #print(batch_src)
            #exit()
            src = [tokenizer.convert_tokens_to_ids(s) for s in batch_src]
            maxlen = max([len(s) for s in batch_src])

            src_mask = []
            padded_sents = []
            for s in src:
                new_s = s + [0] * (maxlen - len(s))
                padded_sents.append(new_s)
                mask = [1] * len(s) + [0] * (maxlen - len(s))
                src_mask.append(mask)
            # B T
            src = torch.tensor(padded_sents).long().cuda()
            # B T
            src_mask = torch.tensor(src_mask).byte().cuda()
            # src, src_mask = prepare_src(train_batch.src, srcpadid)
            tgt = prepare_tgt(train_batch.tgt)
            tag = train_batch.tag

            loss = model(src, src_mask, tgt, tag)

            # "update parameters"


            if decay > 1:
                loss = loss / decay

            loss.backward()

            # if args.grad_clip:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if (iters + 1) % decay == 0:
                opt.step()
                scheduler.step()  # Update learning rate schedule
                opt.zero_grad()

            # opt.step()

            t2 = time.time()

            loss = loss.item()

            print("epoch:{} iters:{} src:({},{}) tgt:({},{}) "
                  "loss:{:.2f} t:{:.2f}".format(epoch + 1, iters + 1, *src.size(),
                                                                       *tgt.size(), loss, t2-t1))

        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        if (epoch + 1) % 1 == 0:
            print("=============validate model==============")
            with torch.no_grad():
                dev.init_epoch()
                model.eval()
                # model.constrain_transition()
                sents = []
                cy_true = []
                cy_pred = []
                for j, dev_batch in enumerate(dev):
                    t1 = time.time()
                    # src, src_mask = prepare_src(dev_batch.src, srcpadid)
                    batch_src = dev_batch.src
                    src = [tokenizer.convert_tokens_to_ids(s) for s in batch_src]
                    maxlen = max([len(s) for s in batch_src])

                    src_mask = []
                    padded_sents = []
                    for s in src:
                        new_s = s + [0] * (maxlen - len(s))
                        padded_sents.append(new_s)
                        mask = [1] * len(s) + [0] * (maxlen - len(s))
                        src_mask.append(mask)
                    # B T
                    src = torch.tensor(padded_sents).long().cuda()
                    # B T
                    src_mask = torch.tensor(src_mask).byte().cuda()

                    tgt = prepare_tgt(dev_batch.tgt)
                    tag = dev_batch.tag.squeeze(-1)
                    _, pre_tag = model.component_extraction(src, src_mask)
                    pre_ctag = model.simile_classify(src, src_mask)
                    cy_true.extend(tag.tolist())
                    cy_pred.extend(pre_ctag.tolist())

                    for sen, tags, p_tags, c_tags in zip(src, tgt, pre_tag, tag):
                        sen = sen[:len(p_tags)].tolist()
                        tags = tags[:len(p_tags)].tolist()
                        if c_tags == 1:
                            sents.append([sen, [tgt_field.vocab.itos[t] for t in tags],
                                        [tgt_field.vocab.itos[t] for t in p_tags]])
                    print('dev iters: {}, t:{}'.format(j, time.time() - t1))

                _, eprecision, erecall, ef1 = evaluate(sents)

                cprecision = precision_score(cy_true, cy_pred)
                crecall = recall_score(cy_true, cy_pred)
                cf1 = f1_score(cy_true, cy_pred)

                print('epoch: {} classify--> precision: {} recall: {} f1: {} best:{}'.format(epoch + 1, cprecision,
                                                                                     crecall, cf1, best_c))
                print('extractor--> precision: {} recall: {} f1: {} best: {}'.format(eprecision, erecall,
                                                                                     ef1, best_e))

                if cf1 > best_c:
                    best_c = cf1
                    best_epoch_for_c = epoch + 1

                    print('save best classifier model at epoch={}'.format(epoch + 1))
                    checkpoint = {'model': model.state_dict(),
                                  'optim': opt.state_dict(),
                                  'args': args }
                    torch.save(checkpoint, '{}/{}.classify.best.pt'.format(args.model_path, args.model))
                    patience_c = 0
                else:
                    patience_c += 1

                if ef1 > best_e:
                    best_e = ef1
                    best_epoch_for_e = epoch + 1

                    print('save best extractor model at epoch={}'.format(epoch + 1))
                    checkpoint = {'model': model.state_dict(),
                                'optim': opt.state_dict(),
                                'args': args
                                }
                    torch.save(checkpoint, '{}/{}.extractor.best.pt'.format(args.model_path, args.model))
                    patience_e = 0
                else:
                    patience_e += 1

        if patience_c > args.patience and patience_e > args.patience:
            print("early stop at {}".format(epoch))
            break

        if args.decay:
            opt.param_groups[0]['lr'] = opt.param_groups[0]['lr'] * args.decay

    print('*******Done********{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    minutes = (time.time() - start) // 60
    if minutes < 60:
        print('best_c:{}, best_e:{} best_epoch_c:{}, best_epoch_e:{}, time:{} mins'.format(best_c, best_e,
                                                                best_epoch_for_c, best_epoch_for_e,
                                                                minutes))
    else:
        hours = minutes / 60
        print('best_c:{}, best_e:{} best_epoch_c:{}, best_epoch_e:{}, time:{:.1f} hours'.format(best_c, best_e,
                                                                     best_epoch_for_c, best_epoch_for_e,
                                                                     hours))


    print('*******Testing************')
    model1 = Classify_Extractor(args,tgt_field)
    model1.cuda()
    load_from = '{}/{}.classify.best.pt'.format(args.model_path, args.model)
    print('load the best model {}'.format(load_from))
    checkpoint = torch.load(load_from, map_location='cpu')
    print('load parameters')
    model1.load_state_dict(checkpoint['model'])

    model2 = Classify_Extractor(args,tgt_field)
    model2.cuda()
    load_from = '{}/{}.extractor.best.pt'.format(args.model_path, args.model)
    print('load the best model {}'.format(load_from))
    checkpoint = torch.load(load_from, map_location='cpu')
    print('load parameters')
    model2.load_state_dict(checkpoint['model'])
    with torch.no_grad():
        test.init_epoch()
        model1.eval()
        model2.eval()
        sents = []
        cy_true = []
        cy_pred = []
        for j, test_batch in enumerate(test):
            t1 = time.time()
            # src, src_mask = prepare_src(test_batch.src, srcpadid)
            batch_src = test_batch.src
            src = [tokenizer.convert_tokens_to_ids(s) for s in batch_src]
            maxlen = max([len(s) for s in batch_src])

            src_mask = []
            padded_sents = []
            for s in src:
                new_s = s + [0] * (maxlen - len(s))
                padded_sents.append(new_s)
                mask = [1] * len(s) + [0] * (maxlen - len(s))
                src_mask.append(mask)
            # B T
            src = torch.tensor(padded_sents).long().cuda()
            # B T
            src_mask = torch.tensor(src_mask).byte().cuda()

            tgt = prepare_tgt(test_batch.tgt)
            tag = test_batch.tag.squeeze(-1)
            _, pre_tag = model2.component_extraction(src, src_mask)
            pre_ctag = model1.simile_classify(src, src_mask)
            cy_true.extend(tag.tolist())
            cy_pred.extend(pre_ctag.tolist())

            # for sen, tags, p_tags in zip(src, tgt, pre_tag):
            #     sen = sen[:len(p_tags)].tolist()
            #     tags = tags[:len(p_tags)].tolist()
            #     sents.append([sen, [tgt_field.vocab.itos[t] for t in tags],
            #                  [tgt_field.vocab.itos[t] for t in p_tags]])
            for sen, tags, p_tags, c_tags in zip(src, tgt, pre_tag, pre_ctag):
                sen = sen[:len(p_tags)].tolist()
                tags = tags[:len(p_tags)].tolist()
                if c_tags == 1:
                    sents.append([sen, [tgt_field.vocab.itos[t] for t in tags],
                                  [tgt_field.vocab.itos[t] for t in p_tags]])
                elif c_tags == 0:
                    sents.append([sen, [tgt_field.vocab.itos[t] for t in tags],
                                  ['O' for t in p_tags]])

            print('test iters: {}, t:{}'.format(j, time.time() - t1))

        _, eprecision, erecall, ef1 = evaluate(sents)

        cprecision = precision_score(cy_true, cy_pred)
        crecall = recall_score(cy_true, cy_pred)
        cf1 = f1_score(cy_true, cy_pred)

        print('Testing classify--> precision: {} recall: {} f1: {}'.format(cprecision, crecall, cf1))
        print('extractor--> precision: {} recall: {} f1: {}'.format(eprecision, erecall, ef1))


