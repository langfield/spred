""" Dataset classes for GPST preprocessing. """
import copy
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

DEBUG = True
SAMPLE = True


class GPSTDataset(Dataset):
    """ Dataset class for GPST (training). """
    def __init__(self, corpus_path, seq_len, encoding="utf-8", on_memory=True):

        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_path = corpus_path
        self.encoding = encoding

        if SAMPLE:
            self.raw_data = pd.read_csv("sample_data.csv")
            self.tensor_data = np.array(self.raw_data.iloc[:, :].values)
        else:
            # Load samples into memory from file.
            self.raw_data = pd.read_csv(corpus_path)

            # Add and adjust columns.
            self.raw_data["Average"] = (
                self.raw_data["High"] + self.raw_data["Low"]
            ) / 2
            self.raw_data["Volume"] = self.raw_data["Volume"] + 0.000001  # Avoid NaNs
            self.raw_data["Average_ld"] = np.log(self.raw_data["Average"]) - np.log(
                self.raw_data["Average"]
            ).shift(1)
            self.raw_data["Volume_ld"] = np.log(self.raw_data["Volume"]) - np.log(
                self.raw_data["Volume"]
            ).shift(1)
            self.raw_data = self.raw_data[1:]

            # Convert data to tensor of shape(rows, features).
            # pylint: disable=not-callable
            self.tensor_data = torch.tensor(self.raw_data.iloc[:, [7, 8]].values)

        self.features = self.create_features(self.tensor_data)
        print("len of features:", len(self.features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    def create_features(self, tensor_data):
        """
        Returns a list of features of the form
        (input, input_raw, is_masked, target, seg_id, label).
        """
        original_data_len = tensor_data.shape[0]
        seq_len = self.seq_len

        if DEBUG:
            print("original_data_len", original_data_len)
            print("seq_len", seq_len)

        num_seqs = original_data_len // seq_len
        input_ids_all = np.arange(0, num_seqs * seq_len)

        features = []
        for i in range(num_seqs):
            inputs_raw = tensor_data[i * seq_len : (i + 1) * seq_len]
            input_ids = input_ids_all[i * seq_len : (i + 1) * seq_len]
            position_ids = np.arange(0, seq_len)
            lm_labels = copy.deepcopy(input_ids)
            targets_raw = copy.deepcopy(inputs_raw)
            features.append(
                (input_ids, position_ids, lm_labels, inputs_raw, targets_raw)
            )

        return features


class GPSTEvalDataset(Dataset):
    """ Dataset class for GPST (evaluation). """
    def __init__(self, tensor_data, seq_len, encoding="utf-8", on_memory=True):

        self.seq_len = seq_len

        self.on_memory = on_memory
        self.encoding = encoding
        self.tensor_data = tensor_data
        self.features = self.create_features(self.tensor_data)
        print("len of features:", len(self.features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    def create_features(self, tensor_data):
        """
        Returns a list of features of the form
        (input, input_raw, is_masked, target, seg_id, label).
        """
        original_data_len = tensor_data.shape[0]
        seq_len = self.seq_len

        if DEBUG:
            print("original_data_len", original_data_len)
            print("seq_len", seq_len)

        num_seqs = original_data_len // seq_len
        input_ids_all = np.arange(0, num_seqs * seq_len)

        features = []
        for i in range(num_seqs):
            inputs_raw = tensor_data[i * seq_len : (i + 1) * seq_len]
            input_ids = input_ids_all[i * seq_len : (i + 1) * seq_len]
            position_ids = np.arange(0, seq_len)
            lm_labels = copy.deepcopy(input_ids)
            targets_raw = copy.deepcopy(inputs_raw)
            features.append(
                (input_ids, position_ids, lm_labels, inputs_raw, targets_raw)
            )

        return features
