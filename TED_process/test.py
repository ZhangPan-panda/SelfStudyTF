# -*- encoding: utf-8 -*-
'''
@File    : test.py
@Time    : 2020/1/22 22:17
@Author  : zp
@Description  : python 3.7 tf 2.0
'''

import seq2seq_train
from seq2seq_train import tf
# import tensorflow as tf
# seq2seq_train.main()
def print_dataset(data_set):
    iterator = data_set.make_one_shot_iterator()
    next_element = iterator.get_next()
    num_batch = 0
    with tf.train.MonitoredTrainingSession() as sess:
        while not sess.should_stop():
            value = sess.run(next_element)
            num_batch += 1
            print("Num Batch: ", num_batch)
            print("Batch value: ", value)

d = seq2seq_train.MakeDataset(seq2seq_train.SRC_TRAIN_DATA)
print_dataset(data_set=d)