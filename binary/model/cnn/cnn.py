from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
if '../../../embeddings' not in sys.path:
    sys.path.append('../../../embeddings')

from seq2tensor import s2t
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, merge
from keras.layers.core import Flatten, Reshape
from keras.layers.merge import Concatenate, concatenate, subtract, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D

from keras.optimizers import Adam,  RMSprop

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.2):
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

# Note: if you use another PPI dataset, this needs to be changed to a corresponding dictionary file.
id2seq_file = '../../../yeast/preprocessed/protein.dictionary.tsv'

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

seq_size = 2000
emb_files = ['../../../embeddings/default_onehot.txt', '../../../embeddings/string_vec5.txt', '../../../embeddings/CTCoding_onehot.txt', '../../../embeddings/vec5_CTC.txt']
use_emb = 0
hidden_dim = 25
n_epochs=50

# ds_file, label_index, rst_file, use_emb, hiddem_dim
ds_file = '../../../sun/preprocessed/Supp-AB.tsv'
label_index = 2
rst_file = 'results/15k_onehot_cnn.txt'
sid1_index = 0
sid2_index = 1
if len(sys.argv) > 1:
    ds_file, label_index, rst_file, use_emb, hidden_dim, n_epochs = sys.argv[1:]
    label_index = int(label_index)
    use_emb = int(use_emb)
    hidden_dim = int(hidden_dim)
    n_epochs = int(n_epochs)

seq2t = s2t(emb_files[use_emb])

max_data = -1
limit_data = max_data > 0
raw_data = []
skip_head = True
x = None
count = 0

for line in tqdm(open(ds_file)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').split('\t')
    if id2index.get(line[sid1_index]) is None or id2index.get(line[sid2_index]) is None:
        continue
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

len_m_seq = np.array([len(line.split()) for line in seq_array])
avg_m_seq = int(np.average(len_m_seq)) + 1
max_m_seq = max(len_m_seq)
print (avg_m_seq, max_m_seq)

dim = seq2t.dim
seq_tensor = np.array([seq2t.embed_normalized(line, seq_size) for line in tqdm(seq_array)])

seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])

print(seq_index1[:10])

class_map = {'0':1,'1':0}
print(class_map)
class_labels = np.zeros((len(raw_data), 2))
for i in range(len(raw_data)):
    class_labels[i][class_map[raw_data[i][label_index]]] = 1.

def build_model():
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    l1=Conv1D(hidden_dim, 3)
    l2=Conv1D(hidden_dim, 3)
    l3=Conv1D(hidden_dim, 3)
    l4=Conv1D(hidden_dim, 3)
    l5=Conv1D(hidden_dim, 3)
    l6=Conv1D(hidden_dim, 3)
    s1=GlobalAveragePooling1D()(l6(MaxPooling1D(2)(l5(MaxPooling1D(2)(l4(MaxPooling1D(2)(l3(MaxPooling1D(2)(l2(MaxPooling1D(2)(l1(seq_input1))))))))))))
    s2=GlobalAveragePooling1D()(l6(MaxPooling1D(2)(l5(MaxPooling1D(2)(l4(MaxPooling1D(2)(l3(MaxPooling1D(2)(l2(MaxPooling1D(2)(l1(seq_input2))))))))))))
    merge_text = multiply([s1, s2])
    x = Dense(hidden_dim, activation='linear')(merge_text)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(int((hidden_dim+7)/2), activation='linear')(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(2, activation='softmax')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model

batch_size1 = 512
adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-5)
rms = RMSprop(lr=0.001)

from sklearn.model_selection import KFold, ShuffleSplit
kf = KFold(n_splits=5, shuffle=True)
tries = 3
cur = 0
recalls = []
accuracy = []
total = []
total_truth = []
train_test = []
for train, test in kf.split(class_labels):
    if np.sum(class_labels[train], 0)[0] > 0.8 * len(train) or np.sum(class_labels[train], 0)[0] < 0.2 * len(train):
        continue
    train_test.append((train, test))
    cur += 1
    if cur >= tries:
        break

print (len(train_test))

#copy below
num_hit = 0.
num_total = 0.
num_pos = 0.
num_true_pos = 0.
num_false_pos = 0.
num_true_neg = 0.
num_false_neg = 0.

for train, test in train_test:
    merge_model = None
    merge_model = build_model()
    adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-6)
    rms = RMSprop(lr=0.001)

    merge_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    merge_model.fit([seq_tensor[seq_index1[train]], seq_tensor[seq_index2[train]]], class_labels[train], batch_size=batch_size1, epochs=n_epochs)
    #result1 = merge_model.evaluate([seq_tensor1[test], seq_tensor2[test]], class_labels[test])
    pred = merge_model.predict([seq_tensor[seq_index1[test]], seq_tensor[seq_index2[test]]])
    for i in range(len(class_labels[test])):
        num_total += 1
        if np.argmax(class_labels[test][i]) == np.argmax(pred[i]):
            num_hit += 1
        if class_labels[test][i][0] > 0.:
            num_pos += 1.
            if pred[i][0] > pred[i][1]:
                num_true_pos += 1
            else:
                num_false_neg += 1
        else:
            if pred[i][0] > pred[i][1]:
                num_false_pos += 1
            else:
                num_true_neg += 1
    accuracy = num_hit / num_total
    prec = num_true_pos / (num_true_pos + num_false_pos)
    recall = num_true_pos / num_pos
    spec = num_true_neg / (num_true_neg + num_false_neg)
    f1 = 2. * prec * recall / (prec + recall)
    mcc = (num_true_pos * num_true_neg - num_false_pos * num_false_neg) / ((num_true_pos + num_true_neg) * (num_true_pos + num_false_neg) * (num_false_pos + num_true_neg) * (num_false_pos + num_false_neg)) ** 0.5
    print (accuracy, prec, recall, spec, f1, mcc)

accuracy = num_hit / num_total
prec = num_true_pos / (num_true_pos + num_false_pos)
recall = num_true_pos / num_pos
spec = num_true_neg / (num_true_neg + num_false_neg)
f1 = 2. * prec * recall / (prec + recall)
mcc = (num_true_pos * num_true_neg - num_false_pos * num_false_neg) / ((num_true_pos + num_true_neg) * (num_true_pos + num_false_neg) * (num_false_pos + num_true_neg) * (num_false_pos + num_false_neg)) ** 0.5
print (accuracy, prec, recall, f1)

with open(rst_file, 'w') as fp:
    fp.write('acc=' + str(accuracy) + '\tprec=' + str(prec) + '\trecall=' + str(recall) + '\tspec=' + str(spec) + '\tf1=' + str(f1) + '\tmcc=' + str(mcc))
