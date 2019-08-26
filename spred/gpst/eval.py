""" Evaluate a trained GPST model and graph its predictions. """
import os
import csv
import copy
import random
import argparse

from typing import Tuple, List, Any

import numpy as np
import pandas as pd

import torch

try:
    from torch.autograd import Variable
except ImportError:
    pass

from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME

from plot import graph
from arguments import get_args
from termplt import plot_to_terminal
from dataset import aggregate, stationarize, normalize, seq_normalize
from modeling_openai import OpenAIGPTLMHeadModel, OpenAIGPTConfig


DEBUG = False
TERM_PRINT = False
# pylint: disable=no-member


def load_model(
    device=None, weights_name: str = WEIGHTS_NAME, config_name: str = CONFIG_NAME
) -> OpenAIGPTLMHeadModel:
    """Load in our pretrained model."""
    # HARDCODE
    output_dir = "checkpoints/"
    output_model_file = os.path.join(output_dir, weights_name)
    output_config_file = os.path.join(output_dir, config_name)
    loaded_config = OpenAIGPTConfig.from_json_file(output_config_file)
    model = OpenAIGPTLMHeadModel(loaded_config)
    model.load_state_dict(torch.load(output_model_file))

    # Set the model to evaluation mode
    if torch.__version__[:5] == "0.3.1":
        model.cuda()
    else:
        model.to(device)
    model.eval()
    print("Is model training:", model.training)
    return model


def get_model(args: argparse.Namespace) -> Tuple[Any]:
    weights_name = args.model_name + ".bin"
    config_name = args.model_name + ".json"

    if torch.__version__[:5] == "0.3.1":
        model = load_model(weights_name=weights_name, config_name=config_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(device, weights_name=weights_name, config_name=config_name)

    dim = model.config.vocab_size
    max_seq_len = model.config.n_positions
    batch_size = args.eval_batch_size
    data_filename = args.dataset
    graph_path = args.graph_dir
    print("Data dimensionality:", dim)
    print("Max sequence length :", max_seq_len)
    print("Eval batch size:", batch_size)

    return (model, dim, max_seq_len, batch_size, data_filename, graph_path)


def load_from_file(
    data_filename, max_seq_len, stat=True, agg_size=1, norm=False, debug=False
) -> pd.DataFrame:
    """
    Returns a dataframe containing the data from `data_filename`
    `stat`: Whether to stationarize the data
    `agg_size`: Size of aggregation bucket (1 means no aggregation)
    `norm`: Whether to normalize the data
    `debug`: Enables logging
    """
    # Grab training data.
    raw_data = pd.read_csv(data_filename, sep="\t")
    if stat:
        if debug:
            print("raw")
            print(raw_data.head(30))

        raw_data = stationarize(raw_data)

        if debug:
            print("stationary")
            print(raw_data.head(30))

    # aggregate the price data to reduce volatility
    raw_data = aggregate(raw_data, agg_size)

    if debug:
        print("aggregate")
        print(raw_data.head(30))

    raw_data = raw_data[1:]
    raw_data = np.array(raw_data)

    # Normalize entire dataset and save scaler object.
    if norm:
        if debug:
            print("Normalizing...")
        raw_data = normalize(raw_data)
        if debug:
            print("Done normalizing.")

    assert len(raw_data) >= max_seq_len

    return raw_data


def predict(
    model: OpenAIGPTLMHeadModel,
    raw_data: pd.DataFrame,
    max_seq_len: int,
    dim: int,
    batch_size: int = 1,
    debug: bool = False,
    seq_norm: bool = False,
) -> np.ndarray:
    """
    Returns
    -------
    ``pred`` shape: <scalar>.
    ``pred`` is the last prediction in the first (and only) batch.
    """
    if torch.__version__[:5] != "0.3.1":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_data = np.array(raw_data)

    if seq_norm:
        tensor_data = seq_normalize(tensor_data)[0]

    tensor_data = torch.Tensor(tensor_data)
    inputs_raw = tensor_data.contiguous()

    # Create ``position_ids``.
    position_ids = torch.arange(0, tensor_data.shape[0])
    position_ids = torch.stack([position_ids])

    # Create ``input_ids``.
    input_ids = copy.deepcopy(position_ids)

    # Reshape.
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
    all_in: np.ndarray, all_out: np.ndarray, graph_path: np.ndarray, data_filename: str
) -> None:
    def matplot(
        graphs_path: str,
        data_filename: str,
        dfs: List[pd.DataFrame],
        ylabels: List[str],
        column_counts: List[int],
    ) -> None:
        """ Do some path handling and call the ``graph()`` function. """
        assert os.path.isdir(graphs_path)
        filename = os.path.basename(data_filename)
        filename_no_ext = filename.split(".")[0]
        save_path = os.path.join(graphs_path, filename_no_ext + ".svg")
        graph(dfs, ylabels, filename_no_ext, column_counts, None, save_path)
        print("Graph saved to:", save_path)

    # Add a second dimension so we can concatenate along it.
    all_in = np.reshape(all_in, (all_in.shape[0], 1))
    all_out = np.reshape(all_out, (all_out.shape[0], 1))
    assert all_out.shape == all_in.shape
    diff = np.concatenate((all_out, all_in), axis=1)

    # ``diff`` shape: (args.width, 2).
    df = pd.DataFrame(diff)
    df.columns = ["pred", "actual"]
    print(df)

    # Format input for ``plot.graph`` function.
    dfs = [df]
    y_label = "Predictions vs Input"
    column_counts = [2]
    matplot(graph_path, data_filename, dfs, y_label, column_counts)


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

    # load model and data
    model, dim, max_seq_len, batch_size, data_filename, graph_path = get_model(args)
    raw_data = load_from_file(
        data_filename,
        max_seq_len,
        args.stationarize,
        args.aggregation_size,
        args.normalize,
    )

    output_list = []
    all_inputs = []
    all_outputs = []

    # Iterate in step sizes of 1 over ``raw_data``.
    start = random.randint(0, len(raw_data) // 2)
    for i in range(start, start + args.width):
        assert i + max_seq_len <= len(raw_data)
        pred = predict(
            model, raw_data[i : i + max_seq_len, :], max_seq_len, dim, batch_size
        )

        # Get the next value in the sequence, i.e., the value we want to predict.
        actual = raw_data[i + max_seq_len, 0]

        if TERM_PRINT:
            output_list = term_print(args, output_list, pred)

        # Append scalar arrays to lists.
        all_outputs.append(pred)
        all_inputs.append(actual)

    # Stack and cast to ``pd.DataFrame``.
    # ``all_in`` and ``all_out`` shape: (args.width,)
    all_in = np.stack(all_inputs)
    all_out = np.stack(all_outputs)

    gen_plot(all_in, all_out, graph_path, data_filename)


def sanity() -> None:
    """Make predictions on a single sequence."""
    # Set hyperparameters.
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()

    # load model and data
    model, dim, max_seq_len, batch_size, data_filename, graph_path = get_model(args)

    output_list = []
    all_inputs = []
    all_outputs = []

    # Iterate in step sizes of 1 over ``raw_data``.
    start = random.randint(0, 100000 // 2)
    start = start - start % max_seq_len
    with open(data_filename) as csvfile:
        read_csv = csv.reader(csvfile, delimiter="\t")
        seq = []
        count = 0
        seq_count = 0
        last_val = [0]
        for row in read_csv:
            # Skip to the start row of the file
            if count < start:
                count += 1
                continue

            float_row = [float(i) for i in row]
            if len(seq) == max_seq_len and seq_count != 0:
                # Get the next value in the sequence, i.e., the value we just predicted
                # print('actual:',float_row)
                actual = float_row[0] - last_val[0]

                all_inputs.append(actual)

                if seq_count == args.width:
                    break

            seq.append(float_row)
            if len(seq) == max_seq_len + 1:
                seq_df = stationarize(
                    pd.DataFrame(
                        seq,
                        columns=["Open", "High", "Low", "Close", "Volume"],
                        dtype=float,
                    )
                )[1:]
                pred = predict(
                    model, seq_df, max_seq_len, dim, batch_size, seq_norm=args.seq_norm
                )
                print("Prediction {} at time step {}".format(pred, count + 1))
                print(seq_df)
                # Step through the input one row at a time.
                input()
                if TERM_PRINT:
                    output_list = term_print(args, output_list, pred)

                # Append scalar arrays to lists.
                all_outputs.append(pred)
                # print('last:', seq[-1:][0])
                last_val = seq[-1:][0]
                seq = seq[1:][:]
                seq_count += 1

            count += 1

    # Stack and cast to ``pd.DataFrame``.
    # ``all_in`` and ``all_out`` shape: (args.width,)
    all_in = np.stack(all_inputs)
    all_out = np.stack(all_outputs)

    gen_plot(all_in, all_out, graph_path, data_filename)


if __name__ == "__main__":
    # sanity()
    main()
