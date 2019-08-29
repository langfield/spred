""" Dataset classes for GPST preprocessing. """
import sys
import copy
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler, StandardScaler

from torch.utils.data import Dataset

DEBUG = True


def stationarize(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a stationarized version of ``input_df``.

    Parameters
    ----------
    input_df : ``pd.DataFrame``, required.
        A 2-dimensional matrix of input data, where the columns are model features
        and the rows are timesteps.
        Shape: ``(total_data_len, vocab_size)``.

    Returns
    -------
    df : ``pd.DataFrame``.
        Shape: ``(total_data_len - 1, vocab_size)``.
    """
    print("Stationarizing...\n")
    df = copy.deepcopy(input_df)
    columns = df.columns
    for col in columns:
        df[col] = df[col] - df[col].shift(1)

    print("Done stationarizing.")
    return df


def aggregate(input_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Returns an aggregated version of ``input_df`` with bucket size ``k``. """
    print("Aggregating...\n")
    if k == 1:
        return input_df
    df = pd.DataFrame()
    columns = input_df.columns

    for j, col in tqdm(enumerate(columns)):
        agg = []
        for i in range(len(input_df[col]) // k):
            agg.append(input_df[col].iloc[i : i + k].values.sum())
        df[col] = agg

        tqdm.write("Columns aggregated: %d out of %d\r" % (j, len(columns)))
        sys.stdout.flush()
    print("Done aggregating.")
    print("Post-aggregation ``input_df`` shape:", input_df.shape)

    return df


def normalize(input_df: pd.DataFrame) -> pd.DataFrame:
    """ Fits a RobustScaler to ``input_df`` and normalizes the inputs. """
    print("Normalizing...")
    input_array = np.array(input_df)
    scaler = RobustScaler()
    scaler.fit(input_array)
    input_array = scaler.transform(input_array)
    df = pd.DataFrame(input_array)
    print("Done normalizing.")
    return df


def seq_normalize(
    inputs_raw: np.ndarray, targets_raw: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
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
        seq_norm: bool = False,
        train_batch_size: int = 1,
    ) -> None:

        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.normalize = normalization
        self.seq_norm = seq_norm

        assert corpus_path[-4:] == ".csv"
        input_df = pd.read_csv(corpus_path, sep="\t")
        print("Raw ``input_df`` shape:", input_df.shape)

        if stationarization:
            # Stationarize each of the columns.
            input_df = stationarize(input_df)
            # remove the first row values as they will be NaN.
            input_df = input_df[1:]

        # Aggregate the price data to reduce volatility.
        input_df = aggregate(input_df, aggregation_size)

        # Normalize entire dataset and save scaler object.
        if self.normalize:
            input_df = normalize(input_df)

        num_batches = len(input_df) // (train_batch_size * seq_len)
        print("Number of complete batches:", num_batches)
        rows_to_keep = train_batch_size * seq_len * num_batches
        print("Rows to keep:", rows_to_keep)
        self.input_array = np.array(input_df.iloc[:rows_to_keep, :].values)
        print("``self.input_array`` shape:", self.input_array.shape)

        self.features = self.create_features(self.input_array)
        print("len of features:", len(self.features))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    def create_features(
        self, tensor_data: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """ Returns a list of features of the form
        (input, input_raw, is_masked, target, seg_id, label).
        """
        original_data_len = tensor_data.shape[0]
        seq_len = self.seq_len

        # Make sure we didn't truncate away all data via ``rows_to_keep``.
        assert original_data_len > 0

        num_seqs = original_data_len // seq_len
        print(
            "Expected number of sequences "
            + "(``original_data_len`` // ``seq_len``) (%d // %d): %d"
            % (original_data_len, seq_len, num_seqs)
        )
        input_ids_all = np.arange(0, num_seqs * seq_len)

        features = []
        print("Creating features...")
        for i in tqdm(range(num_seqs), position=0, leave=True):
            inputs_raw = tensor_data[i * seq_len : (i + 1) * seq_len]
            input_ids = input_ids_all[i * seq_len : (i + 1) * seq_len]
            position_ids = np.arange(0, seq_len)
            lm_labels = copy.deepcopy(input_ids)
            targets_raw = copy.deepcopy(inputs_raw)
            if self.seq_norm:
                inputs_raw, targets_raw = seq_normalize(inputs_raw, targets_raw)

            features.append(
                (input_ids, position_ids, lm_labels, inputs_raw, targets_raw)
            )
        print("Done creating features.")
        return features
