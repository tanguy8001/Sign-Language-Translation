#same as ipynb file, runnable using slurm


import nltk
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize

data_frame = pd.read_csv('/home/grt/youtube-asl_data/data/tsv_files/new_youtube-asl_v1_1.tsv', sep='\t')

# Load training text samples
split = 'train'
column = 'raw-text'
data_frame = data_frame.loc[data_frame['split'].str.contains(split)]
translation = data_frame[column]
vids = data_frame['vid']

# Using Punkt to tokenize words
translation=translation.astype(str)
sent_tks = [word_tokenize(s.lower()) for s in translation]
tag_res = [nltk.pos_tag(tks) for tks in sent_tks]

joined_tag_res = []
for l in tag_res:
    joined_tag_res.extend(l)

freq_dist = nltk.ConditionalFreqDist(joined_tag_res)

exclude_words = ['was', 'i', 'said', 'aslcaptions.com', '\'s', 'is', 'be', 'are', 'has', 'www.aslcaptions.com', 'did', '\'ve', '\'m', '%', 've', 'r', 'd', '*', 'b', 'ed', 'e.', '[', ']', 'dpan.tv', 'iii', '<', '>', '/i', '']
collect_keys = {'NN', 'NNP', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
filtered_words = {}
for word, freq in freq_dist.items():
    if word in exclude_words: continue
    key_set = set(freq.keys())
    intersect = collect_keys.intersection(key_set)
    if len(intersect) > 0:
        filtered_freq = {}
        for tag in intersect:
            if freq[tag] > 10:
                filtered_freq[tag] = freq[tag]
        if len(filtered_freq) > 0: 
            filtered_words[word] = filtered_freq

# Load GloVe embeddings
# vocab = []
# embeddings = []
# with open('/mnt/workspace/slt_baseline/notebooks/glove/glove.6B.300d.txt', 'r') as f:
#     for line in f:
#         items = line.strip().split(' ')
#         vocab.append(items[0])
#         embeddings.append(np.asarray(items[1:], 'float32'))

import json

with open('ytasl-v1.0/uncased_filtred_VNs.json', 'w') as f:
    json.dump(filtered_words, f)
# Filter cross filter with glove vocabulary
import numpy as np
vocab = []
embeddings = []
with open('/home/grt/GloVe/glove.6B/glove.6B.300d.txt', 'r') as f:
    for line in f:
        items = line.strip().split(' ')
        vocab.append(items[0])
        embeddings.append(np.asarray(items[1:], 'float32'))
VN_dict = json.load(open('ytasl-v1.0/uncased_filtred_VNs.json', 'r'))
VNs = VN_dict.keys()

OOV = 0
OOV_word = []
for vn in VNs:
    if vn not in vocab:
        OOV += 1
        OOV_word.append(vn)
for k in OOV_word:
    stat = VN_dict[k]
    total = 0
    for pos, num in stat.items():
        total += num
    print(total, k)
for k in OOV_word:
    VN_dict.pop(k)

with open('ytasl-v1.0/uncased_filtred_glove_VNs.json', 'w') as f:
    json.dump(VN_dict, f)
# Generate Index word mapping for glove filtered VNs
import json

with open('ytasl-v1.0/uncased_filtred_glove_VNs.json', 'r') as f:
    vn_dict = json.load(f)

vn_words = list(vn_dict.keys())
with open('ytasl-v1.0/uncased_filtred_glove_VN_idxs.txt', 'w') as f:
    for idx, word in enumerate(vn_words):
        f.write(f'{idx} {word}\n')
# Generate the corresponding embedding pkl
import numpy as np

vn_glove_embeddings = []

glove_embedding_dict = {}

with open('/home/grt/GloVe/glove.6B/glove.6B.300d.txt', 'r') as f:
    for line in f:
        items = line.strip().split(' ')
        glove_embedding_dict[items[0]] = np.asarray(items[1:], 'float32')
        
for word in vn_words:
    vn_glove_embeddings.append(glove_embedding_dict[word])
vn_glove_embed = np.stack(vn_glove_embeddings, axis=0)
vn_glove_embed.shape
import pickle as pkl
with open('ytasl-v1.0/uncased_filtred_glove_VN_embed.pkl', 'wb') as f:
    pkl.dump(vn_glove_embed, f)
# Generate trainning infomation
data_frame = pd.read_csv('/home/grt/youtube-asl_data/data/tsv_files/new_youtube-asl_v1_1.tsv', sep='\t')

# Load training text samples
split = 'train'
column = 'raw-text'
data_frame = data_frame.loc[data_frame['split'].str.contains(split)]
translation = data_frame[column]
translation = translation.astype(str)
vids = data_frame['vid']

VN_dict = json.load(open('ytasl-v1.0/uncased_filtred_glove_VNs.json', 'r'))
VNs = VN_dict.keys()

matched = {}
for vid, trans in zip(vids, translation):
    ref_word_list = word_tokenize(trans)
    matched_words = []
    for ref_word in ref_word_list:
        if ref_word in VNs:
            matched_words.append(ref_word)
    matched[vid] = matched_words

with open('ytasl-v1.0/uncased_filtred_glove_VN_matched_train.json', 'w') as f:
    json.dump(matched, f)