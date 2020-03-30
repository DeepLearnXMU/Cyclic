#!/bin/bash
datapath=/home/lemon/data/Ch-simile/berttok
modelname=cyclic32e-5
domain=classify.train
gpu=1

CUDA_VISIBLE_DEVICES=${gpu} nohup python -u main.py --model ${modelname} \
--corpus_prex $datapath/${domain} --lang sen etag tag \
--valid $datapath/classify.test --pool 2000 \
--test_corpus $datapath/classify.test \
--embdim 50 --hidden 128 128 32 64 \
--maxepoch 100 \
--k 3 \
--patience 6 \
--optimizer bert \
--lr 2e-5 \
--decay 1.0 \
--drop_ratio 0.5 \
--grad_clip 1.0 \
--seed 1 \
--writetrans decoding/${modelname}.devtrans \
--ref /home/lemon/data/iwslt/zh-en/dev2010.en --batch_size 40 --delay 2 \
--vocab $datapath --init_src_vocab pretrained_embeddings.emb --vocab_size 30000 --load_vocab --smoothing 0.1 --share_embed --beam_size 10 \
--max_len 120 --eval_every 1000 --save_every 1000 >${modelname}.train 2>&1 &
#train_id=$!
#wait $train_id

#CUDA_VISIBLE_DEVICES=${gpu} nohup python main.py --mode test --load_from models/${modelname}.8.backup.pt \
#--test $datapath/${domain}.test --ref ${datapath}/${domain}.test.en \
#--writetrans decoding/${domain}.trans --beam_size 4 >${domain}.translog 2>&1 &

