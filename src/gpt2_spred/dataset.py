import numpy as np
import pandas as pd

import copy
import torch
from torch.utils.data import Dataset

from sample_mask_spred import _sample_mask
from split_a_and_b_spred import _split_a_and_b

DEBUG = False
SIN = True

class GPTSpredDataset(Dataset):
    def __init__(self, 
                 corpus_path, 
                 seq_len, 
                 encoding="utf-8", 
                 on_memory=True):

        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding

        if SIN:
            self.raw_data = pd.read_csv('sin.csv')
            self.tensor_data = np.array(self.raw_data.iloc[:,:].values)
        else:
            # Load samples into memory from file.
            self.raw_data = pd.read_csv(corpus_path)

            # Add and adjust columns.
            self.raw_data["Average"] = (self.raw_data["High"] + self.raw_data["Low"])/2
            self.raw_data['Volume'] = self.raw_data['Volume'] + 0.000001 # Avoid NaNs
            self.raw_data["Average_ld"] = (np.log(self.raw_data['Average']) - 
                                        np.log(self.raw_data['Average']).shift(1))
            self.raw_data["Volume_ld"] = (np.log(self.raw_data['Volume']) - 
                                    np.log(self.raw_data['Volume']).shift(1))
            self.raw_data = self.raw_data[1:]

            # convert data to tensor of shape(rows, features)
            self.tensor_data = torch.tensor(self.raw_data.iloc[:,[7,8]].values)
        
        self.features = self.create_features(self.tensor_data)
        print('len of features:', len(self.features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    def create_features(self, tensor_data):
        """
        Returns a list of features of the form 
        (input, input_raw, is_masked, target, seg_id, label).
        """
        original_data_len = self.tensor_data.shape[0]
        seq_len = self.seq_len
        
        if DEBUG:
            print('original_data_len', original_data_len)
            print('seq_len', seq_len)

        num_seqs = original_data_len // seq_len
        input_ids_all = np.arange(0, num_seqs * seq_len)
         
        for i in range(num_seqs):
            inputs_raw = self.tensor_data[i * seq_len: (i + 1) * seq_len]
            input_ids = input_ids_all[i * seq_len: (i + 1) * seq_len]
            lm_labels = copy.deepcopy(input_ids)
            features.append((input_ids, lm_labels, inputs_raw))      
        
        return features
