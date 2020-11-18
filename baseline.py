# -*- coding: utf-8 -*-
# @Date    : 2020/11/18
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : baseline.py
import glob
from xlrd import XLRDError
import os, sys
from tqdm import tqdm

import pandas as pd
import numpy as np
from toolkit4nlp.utils import DataGenerator, pad_sequences
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.layers import *
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.optimizers import *
from sklearn.metrics import accuracy_score

train_paths = glob.glob('/home/mingming.xu/datasets/NLP/onecity/data/train/*')
test_paths = glob.glob('/home/mingming.xu/datasets/NLP/onecity/data/test1/*')

df_label = pd.read_csv('/home/mingming.xu/datasets/NLP/onecity/data/answer_train.csv')
file2label = {}

for i, item in df_label.iterrows():
    file2label[os.path.split(item['filename'])[-1]] = item['label']

all_labels = set(file2label.values())
id2label = {i: label for i, label in enumerate(all_labels)}
label2id = {label: i for i, label in enumerate(all_labels)}


def read_file(path):
    """
    读取数据，包括文件名/列名/第一行内容/对应label
    """
    df = None

    columns = []
    content = []
    # get label
    fname = os.path.split(path)[-1]
    label = file2label.get(fname, None)

    # 先尝试使用read_excel 再尝试red_csv
    try:
        df = pd.read_excel(path)
    except:
        try:
            df = pd.read_csv(path, error_bad_lines=False, encoding='utf8')
        except:
            pass

    if df is not None and len(df) > 0:
        df = df.astype(str)
        columns = list(df.columns)
        content = df.to_numpy()[0].tolist()
    return [label, fname, columns, content]


def load_data(paths):
    data = []
    for path in tqdm(paths):
        data.append(read_file(path))
    return data


train_data = load_data(train_paths)
test_data = load_data(test_paths)

# 模拟复赛，一半标题不能用
only_title = [t for t in train_data if not t[2]]
has_content = [t for t in train_data if t[2]]
np.random.shuffle(has_content)
half = int(len(has_content) * 0.5)

np.random.shuffle(has_content)
half = int(len(has_content) * 0.5)

remove_title = [[t[0], None] + t[2:] for t in has_content[:half]]
new_has_content = remove_title + has_content[half:]
train_data = only_title + new_has_content

maxlen = 256
batch_size = 16
num_hidden_layer = 3  # transformer 层数
model_save_path = 'best_model.weights'

# BERT base
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []

        for is_end, sample in self.get_sample(shuffle):
            label, title, columns, content = sample
            label = label2id[label] if label else None

            if title:
                token_ids, segment_ids = tokenizer.encode(title)
            else:
                token_ids = [tokenizer._token_start_id]
                segment_ids = [0]

            # 有内容则拼接
            if columns:
                for col, cont in zip(columns, content):
                    #                     print('col', col)
                    #                     print('cont',cont)
                    #                     if type(cont) not in (str, list):
                    #                         cont = list(cont)
                    col_tokens = tokenizer.encode(col)[0][1:]
                    cont_tokens = tokenizer.encode(cont)[0][1:]
                    tokens = col_tokens + cont_tokens
                    token_ids += tokens
                    segment_ids += [1] * len(tokens)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

            if is_end or len(batch_token_ids) == self.batch_size:
                batch_token_ids = pad_sequences(batch_token_ids, maxlen=maxlen)
                batch_segment_ids = pad_sequences(batch_segment_ids, maxlen=maxlen)
                batch_labels = pad_sequences(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels

                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


np.random.shuffle(train_data)
n = int(len(train_data) * 0.8)

train, valid = train_data[n:], train_data[n:]

train_generator = data_generator(train, batch_size)
valid_generator = data_generator(valid, batch_size * 2)
test_generator = data_generator(test_data, batch_size)

bert = build_transformer_model(config_path=config_path,
                               checkpoint_path=checkpoint_path,
                               model='bert',
                               num_hidden_layer=num_hidden_layer
                               )

output = Lambda(lambda x: x[:, 0])(bert.output)
output = Dense(len(label2id), activation='softmax')(output)

model = Model(bert.inputs, output)
model.summary()


def evaluate(data):
    preds = []
    ytrues = []
    for x, y in tqdm(data):
        ytrues.extend(y)
        pred = model.predict(x).argmax(-1)
        preds.append(pred)

    ytrues = np.concatenate(ytrues)
    preds = np.concatenate(preds)

    return accuracy_score(ytrues, preds)


class Evaluator(keras.callbacks.Callback):
    def __init__(self, model_save_path='best_model.weights'):
        self.best_acc = 0.
        self.model_save_path = model_save_path

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(valid_generator)
        if acc > self.best_acc:
            self.best_acc = acc
            self.model.save_weights(self.model_save_path)

        print('epoch: {}, acc: {}, best acc: {}'.format(epoch, acc, self.best_acc))


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=AdaBelief(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)


def write_to_file(data=test_generator, save_path='submit_test1.csv'):
    preds = []
    for x, _ in tqdm(data):
        pred = model.predict(x).argmax(-1)
        preds.append(pred)

    preds = np.concatenate(preds)

    with open(save_path, 'w') as f:
        f.write('filename,label\n')
        for sample, pred in zip(test_data, preds):
            fname = 'test1/' + sample[1]
            label = id2label[pred]
            f.write(','.join([fname, label]) + '\n')


if __name__ == '__main__':
    model.fit_generator(train_generator.generator(),
                        steps_per_epoch=len(train_generator),
                        epochs=5)

    model.load_weights(model_save_path)
    evaluate(valid_generator)

    write_to_file(test_generator)
