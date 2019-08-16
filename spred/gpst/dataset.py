""" Dataset classes for GPST preprocessing. """
import copy
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

DEBUG = True


class GPSTDataset(Dataset):
    """ Dataset class for GPST (training). """

    def __init__(
        self,
        corpus_path: str,
        seq_len: int,
        encoding: str = "utf-8",
        on_memory: bool = True,
        no_price_preprocess: bool = False,
        train_batch_size: int = 1,
    ) -> None:

        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_path = corpus_path
        self.encoding = encoding

        assert corpus_path[-4:] == ".csv"
        self.raw_data = pd.read_csv(corpus_path, sep="\t")

        if not no_price_preprocess:
            columns = self.raw_data.columns
            
            # stationarize each of the columns
            print("columns", columns)
            for col in columns:
                if col == "":
                    continue
                # add a small value to avoid dividing by zero
                self.raw_data[col] = np.cbrt(self.raw_data[col]) - np.cbrt(
                    self.raw_data[col]
                ).shift(1)

            # remove the first row values as they will be NaN
            self.raw_data = self.raw_data[1:]

        num_batches = len(self.raw_data) // (train_batch_size * seq_len)
        rows_to_keep = train_batch_size * seq_len * num_batches
        self.tensor_data = np.array(self.raw_data.iloc[:rows_to_keep, :].values)
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

        # Make sure we didn't truncate away all the data when
        # making sure ``batch_size * seq_len`` evenly divides the number of rows.
        if original_data_len <= 0:
            print("========================================")
            print("Is ``args.train_batch_size`` larger than")
            print("``<total_data_len> // seq_len ``?")
            print("========================================")
            assert original_data_len > 0

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
