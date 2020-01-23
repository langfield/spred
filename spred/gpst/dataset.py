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

# pylint: disable=bad-continuation


def stationarize(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a stationarized version of ``input_df``.

    Parameters
    ----------
    input_df : ``pd.DataFrame``, required.
        A 2-dimensional matrix of input data, where the columns are model features
        and the rows are timesteps.
        Shape: ``(input_df.shape[0], vocab_size)``.

    Returns
    -------
    df : ``pd.DataFrame``.
        Shape: ``(input_df.shape[0] - 1, vocab_size)``.
    """
    print("Stationarizing...\n")
    df = copy.deepcopy(input_df)
    columns = df.columns
    for col in columns:
        df[col] = df[col] - df[col].shift(1)

    print("Done stationarizing.")
    return df


def aggregate(input_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Returns an aggregated version of ``input_df`` with bucket size ``k``.

    Parameters
    ----------
    input_df : ``pd.DataFrame``, required.
        A 2-dimensional matrix of input data, where the columns are model features
        and the rows are timesteps.
        Shape: ``(input_df.shape[0], vocab_size)``.

    Returns
    -------
    df : ``pd.DataFrame``.
        Shape: ``(input_df.shape[0], vocab_size)``.
    """
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
    """
    Fits a RobustScaler to ``input_df`` and normalizes the inputs.

    Parameters
    ----------
    input_df : ``pd.DataFrame``, required.
        A 2-dimensional matrix of input data, where the columns are model features
        and the rows are timesteps.
        Shape: ``(input_df.shape[0], vocab_size)``.

    Returns
    -------
    df : ``pd.DataFrame``.
        Shape: ``(input_df.shape[0], vocab_size)``.
    """
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
    """
    Fits a StandardScaler to ``inputs_raw`` and normalizes the inputs, as well
    as ``targets_raw`` if passed.

    Parameters
    ----------
    inputs_raw : ``np.ndarray``, required.
        A 2-dimensional matrix of input data, where the columns are model features
        and the rows are timesteps.
        Shape: ``(seq_len, vocab_size)``.
    targets_raw : ``np.ndarray``, required.
        A 2-dimensional matrix of target data, where the columns are model features
        and the rows are timesteps.
        Shape: ``(seq_len, vocab_size)``.

    Returns
    -------
    inputs_raw : ``np.ndarray``.
        Shape: ``(seq_len, vocab_size)``.
    targets_raw : ``np.ndarray``.
        Shape: ``(seq_len, vocab_size)``.
    """
    # Normalize ``inputs_raw`` and ``targets_raw``.
    scaler = StandardScaler()
    scaler.fit(inputs_raw)
    inputs_raw = scaler.transform(inputs_raw)
    if targets_raw is not None:
        targets_raw = scaler.transform(targets_raw)
    return inputs_raw, targets_raw


class GPSTDataset(Dataset):
    """ 
    Dataset class for GPST (training). 

    Parameters
    ----------
    corpus_path : ``str``.
        Path to csv corpus file.
    seq_len : ``int``.
        Length of transformer sequences.
    input_dim : ``int``.
        Dimension of the source dataset.
    orderbook_depth : ``int``.
        How many levels we model in the orderbook.
    encoding : ``str``.
        Source file encoding.
    on_memory: ``bool``.
        Not implemented. Lazy loading in order to treat large datasets.
    stationarization : ``bool``.
        Stationarize by computing differences.
    aggregation_size : ``int``.
        Number of timesteps to aggregate (1 does nothing).
    normalization : ``bool``.
        Normalize the entire dataset.
    seq_norm : ``bool``.
        Normalize each sequence individually.
    train_batch_size : ``int``.
        Batch size used during training.
    """

    def __init__(
        self,
        corpus_path: str,
        seq_len: int,
        input_dim: int,
        orderbook_depth: int,
        sep: str,
        step_size: int,
        encoding: str = "utf-8",
        on_memory: bool = True,
        stationarization: bool = False,
        aggregation_size: int = 1,
        normalization: bool = False,
        seq_norm: bool = False,
        train_batch_size: int = 1,
    ) -> None:

        self.seq_len = seq_len
        self.step_size = step_size
        self.input_dim = input_dim
        self.depth = orderbook_depth

        self.on_memory = on_memory
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.normalize = normalization
        self.seq_norm = seq_norm

        assert corpus_path[-4:] == ".csv"
        # Shape: (total_data_len, vocab_size).
        input_df = pd.read_csv(corpus_path, sep=sep)
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
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns a list of features of the form
            ``(input, input_raw, is_masked, target, seg_id, label)``.

        Parameters
        ----------
        tensor_data : ``np.ndarray``, required.
            The 2-dimensional input data, after being optionally stationarized,
            aggregated, and/or normalized.
            Shape: (original_data_len, vocab_size).

        Returns
        -------
        features : ``List[Tuple[np.ndarray, ...]]``.
            Shape: (num_seqs, 4).

            Elements
            --------
            input_ids : ``np.ndarray``.
                The index of the corresponding rows in ``inputs_raw`` in
                ``tensor_data``.
                Shape: (seq_len,).
            position_ids : ``np.ndarray``.
                The index of the rows in ``inputs_raw`` relative to the current
                sequence.
                Shape: (seq_len,).
            labels : ``np.ndarray``.
                Labels for conditional distribution computation.
            inputs_raw : ``np.ndarray``.
                A slice of ``tensor_data`` containing a sequence worth of
                training data for the model.
                Shape: (seq_len, vocab_size).
        """

        original_data_len = tensor_data.shape[0]
        seq_len = self.seq_len
        depth = self.depth

        step_size = self.step_size

        # Make sure we didn't truncate away all data via ``rows_to_keep``.
        assert original_data_len > 0

        # To see why this formula is correct, consider the space after
        # the first sequence until the end of the data. This has size
        # ``original_data_len - seq_len``, and within it we cound the number
        # of sequences past the first we can fit by counting the indices of
        # the end of each sequence, inclusive. This is equivalent to taking
        # integer division by the step size. Then we add one to account for
        # the first sequence.
        num_seqs = ((original_data_len - seq_len) // step_size) + 1
        print(
            "Expected number of sequences "
            + "(``original_data_len`` // ``seq_len``) (%d // %d): %d"
            % (original_data_len, seq_len, num_seqs)
        )
        input_ids_all = np.arange(0, num_seqs * seq_len)

        bid_col = 0
        ask_col = int(self.input_dim / 2)

        features = []
        print("Creating features...")

        i = 0
        while i + seq_len <= len(tensor_data):
            inputs_raw = tensor_data[i : i + seq_len]
            input_ids = input_ids_all[i : i + seq_len]
            position_ids = np.arange(0, seq_len)

            # Compute labels.
            bid_delta_indices = 100 * inputs_raw[..., bid_col]
            bid_delta_indices = bid_delta_indices.astype(int)
            bid_delta_indices[bid_delta_indices > depth] = depth
            bid_delta_indices[bid_delta_indices < (-1 * depth)] = -1 * depth
            bid_delta_indices = bid_delta_indices + depth

            ask_delta_indices = 100 * inputs_raw[..., ask_col]
            ask_delta_indices = ask_delta_indices.astype(int)
            ask_delta_indices[ask_delta_indices > depth] = depth
            ask_delta_indices[ask_delta_indices < (-1 * depth)] = -1 * depth
            ask_delta_indices = ask_delta_indices + depth

            # print("bdi:\n", bid_delta_indices)
            # print("adi:\n", ask_delta_indices)

            # These labels give the true index where the set bit should lie in a
            # one-hot vector of shape ``(seq_len, (2 * depth + 1) ** 2)`` which has
            # reshaped from a one-hot matrix of shape
            #   ``(seq_len, (2 * depth + 1) ** 2), (2 * depth + 1) ** 2)``,
            # where the second dimension is the true bid index and the third dimension
            # is the true ask index.
            flat_class_labels = (2 * depth + 1) * bid_delta_indices + ask_delta_indices

            assert flat_class_labels.shape == (seq_len,)

            if self.seq_norm:
                inputs_raw = seq_normalize(inputs_raw)

            features.append((input_ids, position_ids, flat_class_labels, inputs_raw))
            i += step_size

        print("Done creating features.")

        return features
