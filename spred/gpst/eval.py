""" Evaluates a trained GPST model and graphs its predictions. """
import os
import random
import argparse

from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from transformers.configuration_openai import OpenAIGPTConfig

from dataset import GPSTDataset
from arguments import get_args
from plot.plot import graph
from plot.termplt import plot_to_terminal
from modeling_openai import ConditionalGPSTModel


DEBUG = False
TERM_PRINT = False
# pylint: disable=no-member, bad-continuation


def get_model(args: argparse.Namespace) -> ConditionalGPSTModel:
    """
    Load the model specified by ``args.model_name``.

    Parameters
    ----------
    args : ``argparse.Namespace``, required.
        Evaluation arguments. See ``arguments.py``.

    Returns
    -------
    model : ``OpenAIGPTLMHeadModel``.
        The loaded model, set to ``eval`` mode, and loaded onto the relevant
        device.
    """

    weights_name = args.model_name + ".bin"
    config_name = args.model_name + ".json"

    # HARDCODE
    output_dir = "ckpts/"
    output_model_file = os.path.join(output_dir, weights_name)
    output_config_file = os.path.join(output_dir, config_name)
    loaded_config = OpenAIGPTConfig.from_json_file(output_config_file)
    model = ConditionalGPSTModel(loaded_config)
    model.load_state_dict(torch.load(output_model_file))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Is model training:", model.training)

    return model


def load_from_file(
    args: argparse.Namespace, debug: bool = False
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns a dataframe containing the data from ``args.dataset: str``.

    Parameters
    ----------
    args : ``argparse.Namespace``, required.
        Evaluation arguments. See ``arguments.py``.
    debug : ``bool``.
        Enables logging to stdout.

    Returns
    -------
    input_array : ``np.ndarray``.
        The entire dataset located at ``args.dataset``, optionally
        stationarized, aggregated, and/or normalized.
        Shape: (<rows_after_preprocessing>, vocab_size).
    """

    # HARDCODE
    step_size = 1

    # Dataset.
    train_data = GPSTDataset(
        args.dataset,
        args.seq_len,
        args.dim,
        args.orderbook_depth,
        args.sep,
        step_size,
        stationarization=args.stationarize,
        aggregation_size=args.aggregation_size,
        normalization=args.normalize,
        seq_norm=args.seq_norm,
        train_batch_size=args.train_batch_size,
    )
    print("Length of eval dataset:", len(train_data))
    features = train_data.features

    assert len(features) >= 0
    return features


def gen_plot(arrays: List[np.ndarray], graph_dir: str, data_filename: str) -> None:
    """
    Graphs ``actuals_array`` and ``preds_array`` on the same plot, and saves the figure
    as an ``.svg`` file.

    Parameters
    ----------
    actuals_array : ``np.ndarray``, required.
        A 1-dimensional array of actual values.
        Shape: (args.width,).
    preds_array : ``np.ndarray``, required.
        A 1-dimensional array of predicted values.
        Shape: (args.width,).
    graph_dir : ``str``, required.
        Directory to which we save the resultant matplotlib graph.
    data_filename : ``str``, required.
        Filename of the source data, without extension.
    """

    # Add a column dim to ``actuals_array`` and ``preds_array`` so we may concat along it.
    reshaped_arrays = [np.reshape(arr, (arr.shape[0], 1)) for arr in arrays]

    # Shape check.
    shp = reshaped_arrays[0].shape
    for arr in reshaped_arrays:
        assert arr.shape == shp

    # Shape: ``(args.width, 2)``.
    diff = np.concatenate(reshaped_arrays, axis=1)
    df = pd.DataFrame(diff)
    df.columns = ["pred bid", "pred ask", "actual bid", "actual ask"]
    print(df)

    # Format input for ``plot.graph`` function.
    dfs = [df]
    y_labels = ["Predictions v. Input."]
    column_counts = [2]

    # Create path to save ``.svg``, and call ``graph``.
    assert os.path.isdir(graph_dir)
    filename = os.path.basename(data_filename)
    filename_no_ext = filename.split(".")[0]
    save_path = os.path.join(graph_dir, filename_no_ext + ".svg")
    graph(dfs, y_labels, filename_no_ext, column_counts, save_path)
    print("Graph saved to:", save_path)


def term_print(
    args: argparse.Namespace, output_list: List[np.ndarray], preds: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Print ``output_list`` of predictions to the terminal via ``terminalplot``.
    Create ``out_array`` to print graph as it is populated.
    Shape is the number of iterations we've made, up until we hit
    ``width`` iterations, after which the shape is ``(width,)``.
    ``output_list`` is a running list of the ``args.terminal_plot_width``
    most recent outputs from the forward call.

    Parameters
    ----------
    args : ``argparse.Namespace``, required.
        Evaluation arguments. See ``arguments.py``.
    output_list : ``List[np.ndarray]``, required.
        A running list of the ``args.terminal_plot_width`` most recent predictions i
        from the forward call. The ``np.ndarray``s it contains have shape (,).
        Shape: (args.terminal_plot_width,).
    pred : ``np.ndarray``, required.
        The output of ``predict()``, the last prediction for the given
        ``input_array_slice``.
        Shape: (,).

    Returns
    -------
    output_list : ``List[np.ndarray]``.
        The populated list of predictions to be used as input to this function
        on the next iteration.
    """

    output_list.append(preds)
    if len(output_list) >= args.terminal_plot_width:
        output_list = output_list[1:]
    out_array = np.concatenate(
        [np.array([-1.5]), np.stack(output_list), np.array([1.5])]
    )
    os.system("clear")
    plot_to_terminal(out_array)
    return output_list


def prediction_loop(
    args: argparse.Namespace,
    model: ConditionalGPSTModel,
    features: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Loops over the input array and makes predictions on slices of length
    ``args.seq_len``. Also optionally prints running set of outputs to
    the terminal via a ``term_print()`` call.

    Parameters
    ----------
    args : ``argparse.Namespace``, required.
        Evaluation arguments. See ``arguments.py``.
    model : ``OpenAIGPTLMHeadModel``, required.
        The loaded model, set to ``eval`` mode, and loaded onto the relevant device.
    input_array : ``np.ndarray``, required.
        The full array of input rows to predict on, read from file and preprocessed.
        Shape: (<rows_after_preprocessing>, vocab_size).

    Returns
    -------
    actuals_list : ``List[np.ndarray]``.
        List of actual values for relevant predicted timestep. Each ``np.ndarray``
        has shape (,).
    preds_list : ``List[np.ndarray]``.
        List of predictions. Each ``np.ndarray`` has shape (,).
    """

    bsz = args.eval_batch_size
    seq_len = args.seq_len
    depth = args.orderbook_depth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_list: List[np.ndarray] = []
    pred_bids: List[int] = []
    pred_asks: List[int] = []
    actual_bids: List[int] = []
    actual_asks: List[int] = []

    # Iterate in step sizes of 1 over ``input_array``.
    # HARDCODE
    start = random.randint(0, len(features) // 2)
    for i in tqdm(range(start, start + args.width)):

        # Grab the slice of ``input_array`` we wish to predict on.
        input_ids, position_ids, _, inputs_raw = features[i]

        # Make sure there is room to get actual values.
        assert i < len(features) - 1
        actual_labels = features[i + 1][2]
        assert actual_labels.shape == (seq_len,)
        actual = actual_labels[-1]

        actual_bid_index = actual // (2 * depth + 1)
        actual_ask_index = actual % (2 * depth + 1)

        input_ids = torch.LongTensor(input_ids)
        position_ids = torch.LongTensor(position_ids)
        inputs_raw = torch.FloatTensor(inputs_raw)

        # Add a batch dimension.
        input_ids = input_ids.unsqueeze(0).expand(bsz, -1)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        inputs_raw = inputs_raw.unsqueeze(0).expand(bsz, -1, -1)

        input_ids = input_ids.to(device)
        position_ids = position_ids.to(device)
        inputs_raw = inputs_raw.to(device)

        # Shape check.
        assert input_ids.shape == (bsz, seq_len)
        assert position_ids.shape == (bsz, seq_len)
        assert inputs_raw.shape == (bsz, seq_len, args.dim)

        outputs = model(input_ids, position_ids, None, inputs_raw)

        # Get g distribution for each side.
        g_logit_map = outputs[0]
        g_bid = g_logit_map["bid"]
        g_ask = g_logit_map["ask"]

        assert g_bid.shape == (bsz, seq_len, 2 * depth + 1)
        assert g_ask.shape == (bsz, seq_len, 2 * depth + 1, 2 * depth + 1)

        bid_prediction_logits = g_bid[0][-1]
        ask_prediction_logits = g_ask[0][-1]

        # Type: ``torch.LongTensor``.
        # Shape: ``(,)``.
        pred_bid_index_tensor = torch.argmax(bid_prediction_logits)
        pred_bid_index: int = pred_bid_index_tensor.item()
        pred_ask_index_tensor = torch.argmax(ask_prediction_logits[pred_bid_index])
        pred_ask_index: int = pred_ask_index_tensor.item()

        preds = [pred_bid_index, pred_ask_index]
        actuals = [actual_bid_index, actual_ask_index]

        if TERM_PRINT:
            raise NotImplementedError
            # TODO: Make a fork of terminalplot which allows multiple output lines.
            output_list = term_print(args, output_list, preds)

        # Append scalar arrays to lists.
        pred_bids.append(pred_bid_index)
        pred_asks.append(pred_ask_index)
        actual_bids.append(actual_bid_index)
        actual_asks.append(actual_ask_index)

    pred_bid_array = np.array(pred_bids)
    pred_ask_array = np.array(pred_asks)
    actual_bid_array = np.array(actual_bids)
    actual_ask_array = np.array(actual_asks)

    return pred_bid_array, pred_ask_array, actual_bid_array, actual_ask_array


def main() -> None:
    """
    Parse arguments, load model, set hyperparameters, load data, and
    make predictions on randomly chosen sequences whose concatenated length
    adds up to ``args.width``, starting from ``start``, a randomly chosen
    starting index. Generates matplotlib graph as an ``.svg`` file.
    """

    # Set hyperparameters.
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()

    model: ConditionalGPSTModel = get_model(args)

    # Grab config arguments from model.
    args.seq_len = model.config.n_positions
    args.dim = model.config.input_dim
    args.orderbook_depth = model.config.orderbook_depth

    print("Data dimensionality:", args.dim)
    print("Max sequence length :", args.seq_len)
    print("Eval batch size:", args.eval_batch_size)

    features = load_from_file(args, debug=True)

    arrays: List[np.ndarray] = list(prediction_loop(args, model, features))

    gen_plot(arrays, args.graph_dir, args.dataset)


if __name__ == "__main__":
    main()
