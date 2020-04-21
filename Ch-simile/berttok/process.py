
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

f = open("../classify.test.etag")
fs = open("../classify.test.sen")
fo = open("classify.test.etag", "w")

j = 0
for s, t in zip(fs.readlines(), f.readlines()):
    s = s.strip().split(' ')
    t = t.strip().split(' ')
    temp = []
    num = 0
    for idx, word in enumerate(s):
        word = token_s = tokenizer.tokenize(word)
        num += len(word)

        if t[idx] == 'O':
            for c in word:
                temp.append('O')
        elif t[idx] == 'ts':
            if len(word) == 1:
                temp.append(t[idx])
            else:
                temp.append('tb')
                for i in range(len(word)-2):
                    temp.append('tm')
                temp.append('te')
        elif t[idx] == 'tb':
            if len(word) == 1:
                temp.append(t[idx])
            else:
                temp.append('tb')
                for i in range(len(word)-1):
                    temp.append('tm')
        elif t[idx] == 'te':
            if len(word) == 1:
                temp.append(t[idx])
            else:
                for i in range(len(word)-1):
                    temp.append('tm')
                temp.append('te')
        elif t[idx] == 'vs':
                if len(word) == 1:
                    temp.append(t[idx])
                else:
                    temp.append('vb')
                    for i in range(len(word) - 2):
                        temp.append('vm')
                    temp.append('ve')
        elif t[idx] == 'vb':
            if len(word) == 1:
                temp.append(t[idx])
            else:
                temp.append('vb')
                for i in range(len(word) - 1):
                    temp.append('vm')
        elif t[idx] == 've':
            if len(word) == 1:
                temp.append(t[idx])
            else:
                for i in range(len(word) - 1):
                    temp.append('vm')
                temp.append('ve')
        else:
            for i in word:
                temp.append(t[idx])
    j += 1
    if j % 10 == 0:
        print(j)
    if num == len(temp):
        print(' '.join(temp), file=fo)
    else:
        print('wrong!',' '.join(t),' '.join(s),  ' '.join(temp), num,len(temp))
        exit()

f.close()
fs.close()
fo.close()
                
# f = open('../classify.test.sen')
# fo = open('classify.test.idx',"w")
#
# for s in f.readlines():
#     s = s.strip().split(' ')
#     temp = []
#     for idx, w in enumerate(s):
#         for c in list(w):
#             temp.append(str(idx))
#     print(' '.join(temp), file=fo)
