from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
if '../../../embeddings' not in sys.path:
    sys.path.append('../../../embeddings')

from seq2tensor import s2t
import keras


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, merge, add
from keras.layers.core import Flatten, Reshape
from keras.layers.merge import Concatenate, concatenate, subtract, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D

from keras.optimizers import Adam,  RMSprop

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.25):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

import numpy as np
from tqdm import tqdm

from keras.layers import Input, CuDNNGRU
from numpy import linalg as LA
import scipy

# change
id2seq_file = '../../../mtb/preprocessed/SKEMPI_seq.txt'

id2index = {}
seqs = []
index = 0
for line in open(id2seq_file):
    line = line.strip().split('\t')
    id2index[line[0]] = index
    seqs.append(line[1])
    index += 1
seq_array = []
id2_aid = {}
sid = 0

seq_size = 100
emb_files = ['../../../embeddings/default_onehot.txt', '../../../embeddings/string_vec5.txt', '../../../embeddings/CTCoding_onehot.txt', '../../../embeddings/vec5_CTC.txt']
use_emb = 0
hidden_dim = 25
n_epochs=50

# ds_file, label_index, rst_file, use_emb, hiddem_dim
ds_file = '../../../string/preprocessed/protein.actions.15k.tsv'
label_index = 2
rst_file = 'results/15k_onehot_cnn.txt'
sid1_index = 0
sid2_index = 1
use_log = 0
if len(sys.argv) > 1:
    ds_file, label_index, rst_file, use_emb, hidden_dim, n_epochs, use_log = sys.argv[1:]
    label_index = int(label_index)
    use_emb = int(use_emb)
    hidden_dim = int(hidden_dim)
    n_epochs = int(n_epochs)
    use_log = int(use_log)

seq2t = s2t(emb_files[use_emb])

max_data = -1
limit_data = max_data > 0
raw_data = []
raw_ids = []
skip_head = True
x = None
count = 0

for line in tqdm(open(ds_file)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').replace('\t\t','\t').split('\t')
    raw_ids.append((line[sid1_index], line[sid2_index]))
    if id2_aid.get(line[sid1_index]) is None:
        id2_aid[line[sid1_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid1_index]]])
    line[sid1_index] = id2_aid[line[sid1_index]]
    if id2_aid.get(line[sid2_index]) is None:
        id2_aid[line[sid2_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid2_index]]])
    line[sid2_index] = id2_aid[line[sid2_index]]
    raw_data.append(line)
    if limit_data:
        count += 1
        if count >= max_data:
            break
print (len(raw_data))
print (len(raw_data[0]))

len_m_seq = np.array([len(line.split()) for line in seq_array])
avg_m_seq = int(np.average(len_m_seq)) + 1
max_m_seq = max(len_m_seq)
print (avg_m_seq, max_m_seq)

dim = seq2t.dim
seq_tensor = np.array([seq2t.embed_normalized(line, seq_size) for line in tqdm(seq_array)])

seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])

print(seq_index1[:10])


#
all_min, all_max = 99999999, -99999999
score_labels = np.zeros(len(raw_data))
for i in range(len(raw_data)):
    if use_log:
        score_labels[i] = np.log(float(raw_data[i][label_index]))
    else:
        score_labels[i] = float(raw_data[i][label_index])
    if score_labels[i] < all_min:
        all_min = score_labels[i]
    if score_labels[i] > all_max:
        all_max = score_labels[i]
for i in range(len(score_labels)):
    score_labels[i] = (score_labels[i] - all_min) / (all_max - all_min)
print (all_min, all_max)
#

def build_model():
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    l1=Conv1D(hidden_dim, 3)
    r1=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l2=Conv1D(hidden_dim, 3)
    r2=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l3=Conv1D(hidden_dim, 3)
    r3=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l4=Conv1D(hidden_dim, 3)
    r4=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l5=Conv1D(hidden_dim, 3)
    r5=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l6=Conv1D(hidden_dim, 3)
    s1=MaxPooling1D(2)(l1(seq_input1))
    s1=concatenate([r1(s1), s1])
    s1=MaxPooling1D(2)(l2(s1))
    s1=concatenate([r2(s1), s1])
    s1=MaxPooling1D(3)(l3(s1))
    s1=concatenate([r3(s1), s1])
    s1=l6(s1)
    s1=GlobalAveragePooling1D()(s1)
    s2=MaxPooling1D(2)(l1(seq_input2))
    s2=concatenate([r1(s2), s2])
    s2=MaxPooling1D(2)(l2(s2))
    s2=concatenate([r2(s2), s2])
    s2=MaxPooling1D(3)(l3(s2))
    s2=concatenate([r3(s2), s2])
    s2=l6(s2)
    s2=GlobalAveragePooling1D()(s2)
    merge_text = multiply([s1, s2])
    x = Dense(100, activation='linear')(merge_text)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(1, activation='sigmoid')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model

batch_size1 = 32
adam = Adam(lr=0.005, amsgrad=True, epsilon=1e-5)

from sklearn.model_selection import KFold, ShuffleSplit
kf = KFold(n_splits=10)
tries = 5
cur = 0
recalls = []
accuracy = []
total = []
total_truth = []
train_test = []
for train, test in kf.split(score_labels):
    train_test.append((train, test))
    print (train[:10])
    cur += 1
    if cur >= tries:
        break

print (len(train_test))

num_total = 0.
total_mse = 0.
total_mae = 0.
total_cov = 0.

def scale_back(v):
    if use_log:
        return np.exp(v * (all_max - all_min) + all_min)
    else:
        return v * (all_max - all_min) + all_min

fp2 = open('records/pred_record.'+rst_file[rst_file.rfind('/')+1:], 'w')
for train, test in train_test:
    merge_model = build_model()
    adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-6)
    rms = RMSprop(lr=0.001)

    merge_model.compile(optimizer=adam, loss='mse', metrics=['mse', 'mae'])
    merge_model.fit([seq_tensor[seq_index1[train]], seq_tensor[seq_index2[train]]], score_labels[train], batch_size=batch_size1, epochs=n_epochs)
    #result1 = merge_model.evaluate([seq_tensor1[test], seq_tensor2[test]], score_labels[test])
    pred = merge_model.predict([seq_tensor[seq_index1[test]], seq_tensor[seq_index2[test]]])
    this_mae, this_mse, this_cov = 0., 0., 0.
    this_num_total = 0
    for i in range(len(score_labels[test])):
        this_num_total += 1
        diff = abs(score_labels[test][i] - pred[i])
        this_mae += diff
        this_mse += diff**2
    num_total += this_num_total
    total_mae += this_mae
    total_mse += this_mse
    mse = total_mse / num_total
    mae = total_mae / num_total
    this_cov = scipy.stats.pearsonr(np.ndarray.flatten(pred), score_labels[test])[0]
    for i in range(len(test)):
        fp2.write(str(raw_ids[test[i]][sid1_index]) + '\t' + str(raw_ids[test[i]][sid2_index]) + '\t' + str(scale_back(np.ndarray.flatten(pred)[i])) + '\t' + str(scale_back(score_labels[test[i]])) + '\t' + str(np.ndarray.flatten(pred)[i]) + '\n')
    print(str(raw_ids[test[i]][sid1_index]) + '\t' + str(raw_ids[test[i]][sid2_index]) + '\t' + str(scale_back(np.ndarray.flatten(pred)[i])) + '\t' + str(scale_back(score_labels[test[i]])) + '\t' + str(np.ndarray.flatten(pred)[i]))
    total_cov += this_cov
    print (mse, mae, this_cov)
fp2.close()

mse = total_mse / num_total
mae = total_mae / num_total
total_cov /= len(train_test)
print (mse, mae, total_cov)

with open(rst_file, 'w') as fp:
    fp.write('mae=' + str(mae) + '\nmse=' + str(mse) + '\ncorr=' + str(total_cov))
