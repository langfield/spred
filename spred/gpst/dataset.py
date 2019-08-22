""" Dataset classes for GPST preprocessing. """
import sys
import copy
from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset

DEBUG = True


def stationarize(input_data: pd.DataFrame) -> pd.DataFrame:
    """ Returns a stationarized version of ``input_data``. """
    raw_data = copy.deepcopy(input_data)
    columns = raw_data.columns
    print("columns", columns)
    for col in columns:
        raw_data[col] = raw_data[col] - raw_data[col].shift(1)

    return raw_data


def aggregate(input_data: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Returns an aggregated version of ``input_data`` with bucket size ``k`` """
    if k == 1:
        return input_data
    raw_data = pd.DataFrame()
    columns = input_data.columns
    for col in columns:
        agg = []
        for i in range(len(input_data[col]) // k):
            agg.append(input_data[col].iloc[i : i + k].values.sum())
        raw_data[col] = agg
    return raw_data

def normalize(inputs_raw: np.ndarray, targets_raw: np.ndarray = None) -> Tuple[np.ndarray]:
    """ Fits a StandardScaler to ``inputs_raw`` and normalizes the inputs. """
    # Normalize ``inputs_raw`` and ``targets_raw``.
    scaler = StandardScaler()
    scaler.fit(inputs_raw)
    inputs_raw = scaler.transform(inputs_raw)
    if targets_raw is not None:
        targets_raw = scaler.transform(targets_raw)
    return inputs_raw, targets_raw


class GPSTDataset(Dataset):
    """ Dataset class for GPST (training). """

    def __init__(
        self,
        corpus_path: str,
        seq_len: int,
        encoding: str = "utf-8",
        on_memory: bool = True,
        stationarization: bool = False,
        aggregation_size: int = 1,
        normalization: bool = False,
        train_batch_size: int = 1,
    ) -> None:

        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.normalize = normalization

        assert corpus_path[-4:] == ".csv"
        self.raw_data = pd.read_csv(corpus_path, sep="\t")

        if stationarization:
            print("Preprocessing data...")
            # Stationarize each of the columns.
            self.raw_data = stationarize(self.raw_data)

            # remove the first row values as they will be NaN
            self.raw_data = self.raw_data[1:]

        # aggregate the price data to reduce volatility
        print("Aggregating...")
        sys.stdout.flush()
        self.raw_data = aggregate(self.raw_data, aggregation_size)
        print("Done aggregating.")
        sys.stdout.flush()

        num_batches = len(self.raw_data) // (train_batch_size * seq_len)
        rows_to_keep = train_batch_size * seq_len * num_batches
        self.tensor_data = np.array(self.raw_data.iloc[:rows_to_keep, :].values)
        self.features = self.create_features(self.tensor_data)
        print("len of features:", len(self.features))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    def create_features(self, tensor_data):
        """ Returns a list of features of the form
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
        print("Creating features...")
        for i in tqdm(range(num_seqs)):
            inputs_raw = tensor_data[i * seq_len : (i + 1) * seq_len]
            input_ids = input_ids_all[i * seq_len : (i + 1) * seq_len]
            position_ids = np.arange(0, seq_len)
            lm_labels = copy.deepcopy(input_ids)
            targets_raw = copy.deepcopy(inputs_raw)

            if self.normalize:
                inputs_raw, targets_raw = normalize(inputs_raw, targets_raw)

            features.append(
                (input_ids, position_ids, lm_labels, inputs_raw, targets_raw)
            )
        print("Done creating features.")
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
