""" Evaluates a trained GPST model and graphs its predictions. """
import os
import copy
import random
import argparse

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
from torch import nn
from transformers.configuration_openai import OpenAIGPTConfig

from plot.plot import graph
from plot.termplt import plot_to_terminal
from arguments import get_args
from modeling_openai import ConditionalGPSTModel
from dataset import aggregate, stationarize, normalize, seq_normalize


DEBUG = False
TERM_PRINT = False
# pylint: disable=no-member, bad-continuation


def get_models(args: argparse.Namespace) -> ConditionalGPSTModel:
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

    config = OpenAIGPTConfig.from_pretrained(args.gpst_model)
    weights_names[mode] = "%s_%s.bin" % (args.model_name, mode)
    config_names[mode] = "%s_%s.json" % (args.model_name, mode)

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

    data_filename = args.dataset
    stat = args.stationarize
    agg_size = args.aggregation_size
    norm = args.normalize

    # HARDCODE
    step_size = 1
    seq_len = args.seq_len

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


def predict(
    args: argparse.Namespace, model: OpenAIGPTLMHeadModel, input_array_slice: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ----------
    args : ``argparse.Namespace``, required.
        Evaluation arguments. See ``arguments.py``.
    model : ``OpenAIGPTLMHeadModel``, required.
        The loaded model, set to ``eval`` mode, and loaded onto the relevant device.
    input_array_slice : ``np.ndarray``, required.
        One sequence of ``input_array``.
        Shape: (seq_len, vocab_size).
    Returns
    -------
    pred : ``np.ndarray``.
        The last prediction in the first (and only) batch.
        Shape: (,).
    """

    # Grab arguments from ``args``.
    seq_len = args.seq_len
    dim = args.dim
    batch_size = args.eval_batch_size
    seq_norm = args.seq_norm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seq_norm:
        input_array_slice, _target_array_slice = seq_normalize(input_array_slice)

    tensor_data = torch.Tensor(input_array_slice)
    inputs_raw = tensor_data.contiguous()

    # Create ``position_ids``.
    position_ids = torch.arange(0, tensor_data.shape[0])
    position_ids = torch.stack([position_ids])

    # Create ``input_ids``.
    input_ids = copy.deepcopy(position_ids)

    # Reshape to add ``batch_size`` dimension.
    inputs_raw = inputs_raw.view(batch_size, seq_len, dim)
    input_ids = input_ids.view(batch_size, seq_len)
    position_ids = position_ids.view(batch_size, seq_len)

    # Casting to correct ``torch.Tensor`` type.
    input_ids = input_ids.to(device)
    position_ids = position_ids.to(device)
    inputs_raw = inputs_raw.to(device)

    # Shape check.
    assert input_ids.shape == (batch_size, seq_len)
    assert position_ids.shape == (batch_size, seq_len)
    assert inputs_raw.shape == (batch_size, seq_len, dim)

    # ``predictions`` shape: (batch_size, seq_len, dim).
    outputs = model(input_ids, position_ids, None, inputs_raw)
    predictions = outputs[0]

    # Casting to correct ``torch.Tensor`` type.
    pred = np.array(predictions[0, -1].data)

    return pred


def gen_plot(
    actuals_array: np.ndarray,
    preds_array: np.ndarray,
    graph_dir: str,
    data_filename: str,
) -> None:
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
    actuals_array = np.reshape(actuals_array, (actuals_array.shape[0], 1))
    preds_array = np.reshape(preds_array, (preds_array.shape[0], 1))
    assert preds_array.shape == actuals_array.shape

    # Shape: ``(args.width, 2)``.
    diff = np.concatenate((preds_array, actuals_array), axis=1)
    df = pd.DataFrame(diff)
    df.columns = ["pred", "actual"]
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
    args: argparse.Namespace, output_list: List[np.ndarray], pred: np.ndarray
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
    output_list.append(pred)
    if len(output_list) >= args.terminal_plot_width:
        output_list = output_list[1:]
    out_array = np.concatenate(
        [np.array([-1.5]), np.stack(output_list), np.array([1.5])]
    )
    os.system("clear")
    plot_to_terminal(out_array)
    return output_list


def prediction_loop(
    args: argparse.Namespace, model: OpenAIGPTLMHeadModel, features: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
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

    output_list: List[np.ndarray] = []
    actuals_list = []
    preds_list = []

    # Iterate in step sizes of 1 over ``input_array``.
    # HARDCODE
    start = random.randint(0, len(features) // 2)
    for i in range(start, start + args.width):

        # Grab the slice of ``input_array`` we wish to predict on.
        input_ids, position_ids, labels, inputs_raw = features[i]

        # Make sure there is room to get actual values.
        assert i < len(features) - 1
        actuals_raw = features[i + 1]
        actual = actuals_raw[0][-1]

        # Cast data to float tensor.
        # TODO: Is this necessary? What is its type before and after?
        inputs_raw = inputs_raw.float()
        labels = labels.long()

        # Handles lack of batch size data truncation in dataset class.
        if not args.stationarize and input_ids.shape[0] < args.train_batch_size:
            continue

        # Shape check.
        assert input_ids.shape == (bsz, seq_len)
        assert position_ids.shape == (bsz, seq_len)
        assert labels.shape == (bsz, seq_len)
        assert inputs_raw.shape == (bsz, seq_len, args.dim)

        outputs = model(input_ids, position_ids, labels, inputs_raw)

        # Get g distribution for each side.
        g_logit_map = outputs[0]
        g_bid = g_logit_map["bid"]
        g_ask = g_logit_map["ask"]

        assert g_bid.shape = (bsz, seq_len, 2 * depth + 1)
        assert g_ask.shape = (bsz, seq_len, 2 * depth + 1, 2 * depth + 1)

        bid_prediction_logits = g_bid[0][-1]
        ask_prediction_logits = g_ask[0][-1]

        _, bid_index = torch.max(bid_prediction_logits)
        _, ask_index = torch.max(ask_prediction_logits[bid_index])

        raise NotImplementedError

        # Make prediction and get ``actual``: the value we want to predict.
        pred = predict(args, model, input_array_slice)

        # HARDCODE: get ``Close`` price at index ``3``.
        actual = actual[3]
        assert actual.shape == pred.shape

        if TERM_PRINT:
            output_list = term_print(args, output_list, pred)

        # Append scalar arrays to lists.
        actuals_list.append(actual)
        preds_list.append(pred)

    return actuals_list, preds_list


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

    model_dict: nn.ModuleDict = get_models(args)

    # Grab config arguments from model.
    args.seq_len = model.config.n_positions
    args.dim = model.config.input_dim

    print("Data dimensionality:", args.dim)
    print("Max sequence length :", args.seq_len)
    print("Eval batch size:", args.eval_batch_size)

    features = load_from_file(args, debug=True)

    actuals_list, preds_list = prediction_loop(args, model, features)

    # Stack and cast to ``pd.DataFrame``.
    # ``actuals_array`` and ``preds_array`` shape: (args.width,)
    actuals_array = np.stack(actuals_list)
    preds_array = np.stack(preds_list)

    gen_plot(actuals_array, preds_array, args.graph_dir, args.dataset)


if __name__ == "__main__":
    main()
