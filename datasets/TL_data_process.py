#!/usr/bin/python
# -*- coding:utf-8 -*-
from datasets.data_package import dataset
import pandas as pd
from datasets.sequence_aug import *

normlizetype = 'none'

data_transforms = {
    'Src_train': Compose([
        Normalize(normlizetype),
        Retype()
    ]),
    'Tgt_train': Compose([
        Normalize(normlizetype),
        Retype()
    ]),
    'Src_val': Compose([
        Normalize(normlizetype),
        Retype()
    ]),
    'Tgt_test': Compose([
        Normalize(normlizetype),
        Retype()
    ])
}


class TL_Process(object):
    input_channel = 1
    output_channel = 1

    def __init__(self, data_dir, data_file):
        self.data_dir = data_dir
        self.data_file = data_file

    def data_prepare(self, test=False):
        if test:
            test_tgt_pd = pd.read_pickle(self.data_dir + 'Tgt_test_' + self.data_file + '.pkl')
            test_tgt_dataset = dataset(anno_pd=test_tgt_pd, data_dir=self.data_dir, test=True, transform=data_transforms['Tgt_test'])

            return test_tgt_dataset, test_tgt_pd

        else:
            train_src_pd = pd.read_pickle(self.data_dir + 'Src_train_' + self.data_file + '.pkl')
            train_tgt_pd = pd.read_pickle(self.data_dir + 'Tgt_train_' + self.data_file + '.pkl')
            val_tgt_pd = pd.read_pickle(self.data_dir + 'Src_val_' + self.data_file + '.pkl')

            train_src_dataset = dataset(anno_pd=train_src_pd, data_dir=self.data_dir, transform=data_transforms['Src_train'])
            train_tgt_dataset = dataset(anno_pd=train_tgt_pd, data_dir=self.data_dir, transform=data_transforms['Tgt_train'])
            val_src_dataset = dataset(anno_pd=val_tgt_pd, data_dir=self.data_dir, transform=data_transforms['Src_val'])

            return train_src_dataset, train_tgt_dataset, val_src_dataset



