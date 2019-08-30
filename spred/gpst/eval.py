""" Evaluate a trained GPST model and graph its predictions. """
import os
import csv
import copy
import random
import argparse

from typing import List, Tuple

import numpy as np
import pandas as pd

import torch

try:
    from torch.autograd import Variable
except ImportError:
    pass

from plot import graph
from arguments import get_args
from termplt import plot_to_terminal
from dataset import aggregate, stationarize, normalize, seq_normalize
from modeling_openai import OpenAIGPTLMHeadModel, OpenAIGPTConfig


DEBUG = False
TERM_PRINT = False
CHECK_SANITY = False
# pylint: disable=no-member


def get_model(args: argparse.Namespace) -> OpenAIGPTLMHeadModel:
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
    output_dir = "checkpoints/"
    output_model_file = os.path.join(output_dir, weights_name)
    output_config_file = os.path.join(output_dir, config_name)
    loaded_config = OpenAIGPTConfig.from_json_file(output_config_file)
    model = OpenAIGPTLMHeadModel(loaded_config)
    model.load_state_dict(torch.load(output_model_file))

    if torch.__version__[:5] == "0.3.1":
        model.cuda()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()
    print("Is model training:", model.training)

    return model


def load_from_file(args: argparse.Namespace, debug: bool = False) -> np.ndarray:
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

    max_seq_len = args.max_seq_len

    input_df = pd.read_csv(data_filename, sep="\t")
    if debug:
        print("Raw:\n", input_df.head(30))

    if stat:
        input_df = stationarize(input_df)
        if debug:
            print("Stationarized:\n", input_df.head(30))

    input_df = aggregate(input_df, agg_size)
    if debug:
        print("Aggregated:\n", input_df.head(30))

    input_df = input_df[1:]

    if norm:
        input_df = normalize(input_df)

    input_array = np.array(input_df)
    assert len(input_array) >= max_seq_len
    return input_array


def predict(
    args: argparse.Namespace, model: OpenAIGPTLMHeadModel, input_array_slice: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ----------
    args : ``argparse.Namespace``, required.
        Evaluation arguments. See ``arguments.py``.
    model : ``OpenAIGPTLMHeadModel``.
        The loaded model, set to ``eval`` mode, and loaded onto the relevant device.
    input_array_slice : ``np.ndarray``.
        One sequence of ``input_array``.
        Shape: (seq_len, vocab_size).
    Returns
    -------
    pred : ``np.ndarray``.
        The last prediction in the first (and only) batch.
        Shape: (,).
    """
    # Grab arguments from ``args``.
    max_seq_len = args.max_seq_len
    dim = args.dim
    batch_size = args.eval_batch_size
    seq_norm = args.seq_norm

    if torch.__version__[:5] != "0.3.1":
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
    inputs_raw = inputs_raw.view(batch_size, max_seq_len, dim)
    input_ids = input_ids.view(batch_size, max_seq_len)
    position_ids = position_ids.view(batch_size, max_seq_len)

    # Casting to correct ``torch.Tensor`` type.
    if torch.__version__[:5] == "0.3.1":
        input_ids = input_ids.long().cuda()
        position_ids = Variable(position_ids.long().cuda()).contiguous()
        inputs_raw = Variable(inputs_raw.cuda()).contiguous()
    else:
        input_ids = input_ids.to(device)
        position_ids = position_ids.to(device)
        inputs_raw = inputs_raw.to(device)

    # Shape check.
    assert input_ids.shape == (batch_size, max_seq_len)
    assert position_ids.shape == (batch_size, max_seq_len)
    assert inputs_raw.shape == (batch_size, max_seq_len, dim)

    # ``predictions`` shape: (batch_size, max_seq_len, dim).
    outputs = model(input_ids, position_ids, None, inputs_raw)
    predictions = outputs[0]

    # Casting to correct ``torch.Tensor`` type.
    if torch.__version__[:5] == "0.3.1":
        pred = np.array(predictions[0, -1].data)[0]
    else:
        pred = np.array(predictions[0, -1].data)

    return pred


def gen_plot(
    all_in: np.ndarray, all_out: np.ndarray, graph_dir: str, data_filename: str
) -> None:
    """
    Graphs ``all_in`` and ``all_out`` on the same plot, and saves the figure
    as an ``.svg`` file.
    """

    # Add a column dim to ``all_in`` and ``all_out`` so we may concat along it.
    all_in = np.reshape(all_in, (all_in.shape[0], 1))
    all_out = np.reshape(all_out, (all_out.shape[0], 1))
    assert all_out.shape == all_in.shape

    # Shape: ``(args.width, 2)``.
    diff = np.concatenate((all_out, all_in), axis=1)
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
    graph(dfs, y_labels, filename_no_ext, column_counts, None, save_path)
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
    args: argparse.Namespace, model: OpenAIGPTLMHeadModel, input_array: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Parameters
    ----------
    input_array : ``np.ndarray``, required.
    args : ``argparse.Namespace``, required.
    model : ``OpenAIGPTLMHeadModel``, required.

    Returns
    -------
    all_inputs : ``List[np.ndarray``.
    all_outputs : ``List[np.ndarray``.
    """
    output_list: List[np.ndarray] = []
    all_inputs = []
    all_outputs = []

    # Iterate in step sizes of 1 over ``input_array``.
    start = random.randint(0, len(input_array) // 2)
    for i in range(start, start + args.width):

        # Grab the slice of ``input_array`` we wish to predict on.
        assert i + args.max_seq_len <= len(input_array)
        input_array_slice = input_array[i : i + args.max_seq_len, :]
        actual_array_slice = input_array[i + 1 : i + args.max_seq_len + 1, :]
        if args.seq_norm:
            actual_array_slice, _ = seq_normalize(actual_array_slice)
        actual = actual_array_slice[-1]

        # Make prediction and get ``actual``: the value we want to predict.
        pred = predict(args, model, input_array_slice)

        # HARDCODE: get ``Close`` price at index ``3``.
        actual = actual[3]
        assert actual.shape == pred.shape

        if TERM_PRINT:
            output_list = term_print(args, output_list, pred)

        # Append scalar arrays to lists.
        all_inputs.append(actual)
        all_outputs.append(pred)

    return all_inputs, all_outputs


def sanity_loop(
    args: argparse.Namespace, model: OpenAIGPTLMHeadModel, input_array: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Parameters
    ----------
    input_array : ``np.ndarray``, required.
    args : ``argparse.Namespace``, required.
    model : ``OpenAIGPTLMHeadModel``, required.

    Returns
    -------
    all_inputs : ``List[np.ndarray``.
    all_outputs : ``List[np.ndarray``.
    """
    output_list: List[np.ndarray] = []
    all_inputs = []
    all_outputs = []

    # Iterate in step sizes of 1 over ``input_array``.
    # HARDCODE
    start = random.randint(0, 100000 // 2)
    start = start - start % args.max_seq_len
    with open(args.dataset) as csvfile:
        read_csv = csv.reader(csvfile, delimiter="\t")
        seq: List[List[float]] = []
        count = 0
        seq_count = 0
        last_val = [0.0]
        for row in read_csv:
            # Skip to the start row of the file.
            if count < start:
                count += 1
                continue

            float_row = [float(i) for i in row]
            if len(seq) == args.max_seq_len and seq_count != 0:
                # Get the next value in the sequence, i.e., the value we just predicted.
                actual = float_row[0] - last_val[0]

                all_inputs.append(actual)

                if seq_count == args.width:
                    break

            seq.append(float_row)
            if len(seq) == args.max_seq_len + 1:
                seq_df = stationarize(
                    pd.DataFrame(
                        seq,
                        columns=["Open", "High", "Low", "Close", "Volume"],
                        dtype=float,
                    )
                )[1:]
                input_array = np.array(seq_df)
                pred = predict(args, model, input_array)
                print("Prediction {} at time step {}".format(pred, count + 1))
                print(seq_df)
                # Step through the input one row at a time.
                input()
                if TERM_PRINT:
                    output_list = term_print(args, output_list, pred)

                # Append scalar arrays to lists.
                all_outputs.append(pred)
                last_val = seq[-1:][0]
                seq = seq[1:][:]
                seq_count += 1

            count += 1

    return all_inputs, all_outputs


def main() -> None:
    """
    Make predictions on randomly chosen sequences whose concatenated length
    adds up to ``args.width``, starting from ``start``, a randomly chosen
    starting index.
    """
    # Set hyperparameters.
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()

    model = get_model(args)

    # Grab config arguments from model.
    args.max_seq_len = model.config.n_positions
    args.dim = model.config.vocab_size

    print("Data dimensionality:", args.dim)
    print("Max sequence length :", args.max_seq_len)
    print("Eval batch size:", args.eval_batch_size)

    input_array = load_from_file(args, debug=True)

    if CHECK_SANITY:
        all_inputs, all_outputs = sanity_loop(args, model, input_array)
    else:
        all_inputs, all_outputs = prediction_loop(args, model, input_array)

    # Stack and cast to ``pd.DataFrame``.
    # ``all_in`` and ``all_out`` shape: (args.width,)
    all_in = np.stack(all_inputs)
    all_out = np.stack(all_outputs)

    gen_plot(all_in, all_out, args.graph_dir, args.dataset)


if __name__ == "__main__":
    # HARDCODE
    # sanity()
    main()
