import torch
import numpy as np
from search.beam import Beam

def greedy_search(args, model, src, src_mask, bosid, eosid):
    time_dim = 1
    B, L = src.size()

    max_len = src.size(time_dim) * args.length_ratio if args.length_ratio else \
              src.size(time_dim) * 2
    min_len = src.size(time_dim) / 2

    annots, initial_state, mapped_keys = model.encoding(src)
    last_words = torch.LongTensor(B,1).fill_(bosid).to(src.device)
    mask = torch.ones_like(last_words).squeeze(time_dim)
    last_words_emb = model.tgt_emb(last_words).squeeze(time_dim)
    src_mask = src_mask.squeeze(time_dim).transpose(1,0)

    # print(mask.size(), last_words_emb.size(), src_mask.size())

    eos_yet = torch.zeros(B, device=src.device, dtype=torch.uint8)
    outs = torch.LongTensor(B, 1).fill_(eosid).to(src.device)

    next_state = initial_state
    for k in range(max_len):
        next_state, context = model.decoder.step(last_words_emb, mask, next_state, mapped_keys, annots, src_mask)
        log_probs = model.generator.forward(last_words_emb, next_state, context)
        if k < min_len:
            log_probs[:,eosid] = -np.inf
        max_value, max_index = log_probs.max(-1)
        last_words_emb = model.tgt_emb(max_index.unsqueeze(time_dim)).squeeze(time_dim)

        outs = torch.cat([outs, max_index.unsqueeze(1)], -1)

        eos_yet = eos_yet | (max_index == eosid)
        if eos_yet.all():
            break

    return outs[:, 1:]

# args, model, src, src_mask, bosid, eosid
# def beamsearch(model, src, beam_size=10, normalize=False, max_len=None, min_len=None):
def beam_search(args, model, src, key_mask, bos_id, eos_id):
    time_dim = 1

    max_len = src.size(time_dim) * 3
    min_len = src.size(time_dim) / 2

    values, hidden, keys = model.encoding(src)

    prev_beam = Beam(args.beam_size)
    prev_beam.candidates = [[bos_id]]
    prev_beam.scores = [0]
    f_done = (lambda x: x[-1] == eos_id)

    valid_size = args.beam_size

    hyp_list = []
    key_mask = key_mask.transpose(1,0)
    for k in range(max_len):
        candidates = prev_beam.candidates
        input = src.new_tensor(list(map(lambda cand: cand[-1], candidates)))
        input = model.tgt_emb(input)
        mask = input.new_ones(input.shape[0])
        hidden, context = model.decoder.step(input, mask, hidden, keys, values, key_mask)

        log_prob = model.generator(input, hidden, context)

        if k < min_len:
            log_prob[:, eos_id] = -float('inf')
        if k == max_len - 1:
            eos_prob = log_prob[:, eos_id].clone()
            log_prob[:, :] = -float('inf')
            log_prob[:, eos_id] = eos_prob
        next_beam = Beam(valid_size)
        done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
        hyp_list.extend(done_list)
        valid_size -= len(done_list)

        if valid_size == 0:
            break

        beam_remain_ix = src.new_tensor(remain_list)
        values = values.index_select(1, beam_remain_ix)  # select batch dim
        keys = keys.index_select(1, beam_remain_ix)
        key_mask = key_mask.index_select(1, beam_remain_ix)
        hidden = hidden.index_select(0, beam_remain_ix)  # hidden is 2-dim tensor
        prev_beam = next_beam

    score_list = [hyp[1] for hyp in hyp_list]
    hyp_list = [hyp[0][1: hyp[0].index(eos_id)] if eos_id in hyp[0] else hyp[0][1:] for hyp in hyp_list]

    for k, (hyp, score) in enumerate(zip(hyp_list, score_list)):
        if len(hyp) > 0:
            lp = (5 + len(hyp)) / (5 + 1)
            lp = lp ** args.alpha # length norm
            score_list[k] = score_list[k] / lp

    score = hidden.new_tensor(score_list)
    sort_score, sort_ix = torch.sort(score)
    output = []
    for ix in sort_ix.tolist():
        output.append((hyp_list[ix], score[ix].item()))
    # add batch dim
    output = src.new_tensor([output[0][0]])
    return output