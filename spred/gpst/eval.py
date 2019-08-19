import copy
import numpy as np
import pandas as pd
import argparse
import os
import torch
from modeling_openai import OpenAIGPTLMHeadModel, OpenAIGPTConfig
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from termplt import plot_to_terminal
from plot import graph
if torch.__version__[:5] == "0.3.1":
    from torch.autograd import Variable
    from torch_addons.sampler import SequentialSampler
else:
    from torch.utils.data import SequentialSampler

DEBUG = False


def eval_config(parser):
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--input", type=str, default="../exchange/concatenated_price_data/ETHUSDT_drop.csv")
    parser.add_argument("--output_dir", type=str, default="graphs/")
    parser.add_argument("--terminal_plot_width", type=int, default=50)
    parser.add_argument(
        "--stationarize", action="store_true", help="Whether to stationarize the raw data"
    )

    return parser

def load_model(device=None, weights_name: str = WEIGHTS_NAME, config_name: str = CONFIG_NAME) -> OpenAIGPTLMHeadModel:
    """Load in our pretrained model."""
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


def create_sample_data(
    dim: int, max_seq_len: int, width: int, plot: int
) -> torch.Tensor:
    """Construct sample time series."""
    # x vals.
    time = np.arange(0, width, 100 / max_seq_len)
    # y vals.
    # price = np.sin(time) + 10
    price = np.array([0] * max_seq_len)

    if plot:
        plt.plot(time, price)
        plt.title("Sample Time Series")
        plt.xlabel("Time (min)")
        plt.ylabel("Price")
        plt.show()

    df = pd.DataFrame({"Price": price})
    df = df[[col for col in df.columns for i in range(dim)]]
    tensor_data = torch.Tensor(np.array(df))
    return tensor_data


def main() -> None:
    """Make predictions on a single sequence."""
    if torch.__version__[:5] == "0.3.1":
        model = load_model(weights_name=WEIGHTS_NAME)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(device, weights_name=WEIGHTS_NAME)

    # Set hyperparameters.
    parser = argparse.ArgumentParser()
    parser = eval_config(parser)
    args = parser.parse_args()

    DIM = model.config.vocab_size
    MAX_SEQ_LEN = model.config.n_positions
    BATCH_SIZE = args.batch
    DATA_FILENAME = args.input
    GRAPH_PATH = args.output_dir
    print("Data dimensionality:", DIM)
    print("Max sequence length :", MAX_SEQ_LEN)
    print("Eval batch size:", BATCH_SIZE)

    """
    # Get sample data.
    tensor_data = create_sample_data(DIM, MAX_SEQ_LEN, WIDTH, PLOT)
    inputs_raw = tensor_data.contiguous()
    """
    
    # Grab training data.
    raw_data = pd.read_csv(DATA_FILENAME, sep="\t")
    if args.stationarize:
        columns = raw_data.columns
        # stationarize each of the columns
        for col in columns:
            raw_data[col] = np.cbrt(raw_data[col]) - np.cbrt(
                raw_data[col]
            ).shift(1)
        print(raw_data.head())
        raw_data = raw_data[1:]

    assert len(raw_data) >= MAX_SEQ_LEN
    output_list = []
    all_inputs = []
    all_outputs = []

    # Iterate in step sizes of 1 over ``raw_data``.
    # HARDCODE
    for i in range(args.width):
        assert i + MAX_SEQ_LEN <= len(raw_data)
        tensor_data = np.array(raw_data.iloc[i:i + MAX_SEQ_LEN, :].values)
        tensor_data = torch.Tensor(tensor_data)
        inputs_raw = tensor_data.contiguous()

        # Create ``position_ids``.
        position_ids = torch.arange(0, tensor_data.shape[0])
        position_ids = torch.stack([position_ids])

        # Create ``input_ids``.
        input_ids = copy.deepcopy(position_ids)

        # Reshape.
        inputs_raw = inputs_raw.view(BATCH_SIZE, MAX_SEQ_LEN, DIM)
        input_ids = input_ids.view(BATCH_SIZE, MAX_SEQ_LEN)
        position_ids = position_ids.view(BATCH_SIZE, MAX_SEQ_LEN)

        # Casting to correct ``torch.Tensor`` type.
        if torch.__version__[:5] == "0.3.1":
            input_ids = input_ids.long().cuda()
            position_ids = Variable(position_ids.long().cuda()).contiguous()
            inputs_raw = Variable(inputs_raw.cuda()).contiguous()
        else:
            input_ids = input_ids.to(device)
            position_ids = position_ids.to(device)
            inputs_raw = inputs_raw.to(device)

        if DEBUG:
            print("================TYPECHECK==================")
            print("Type of input_ids:", type(input_ids))
            print("Type of position_ids:", type(position_ids))
            if torch.__version__[:5] == "0.3.1":
                print("type of position_ids data:", type(position_ids.data))
            print("Type of inputs_raw:", type(inputs_raw))
            print("================SHAPECHECK=================")
            print("input_ids shape:", input_ids.shape)
            print("position_ids shape:", position_ids.shape)
            print("inputs_raw shape:", inputs_raw.shape)

        # Shape check.
        assert input_ids.shape == (BATCH_SIZE, MAX_SEQ_LEN)
        assert position_ids.shape == (BATCH_SIZE, MAX_SEQ_LEN)
        assert inputs_raw.shape == (BATCH_SIZE, MAX_SEQ_LEN, DIM)

        outputs = model(input_ids, position_ids, None, None, inputs_raw)
        # Shape: (BATCH_SIZE, MAX_SEQ_LEN, DIM)
        # Type: torch.autograd.Variable
        predictions = outputs[0]
        
        # How many time steps fit in terminal window.
        GRAPH_WIDTH = args.terminal_plot_width

        # ``output_list`` is a running list of the ``GRAPH_WIDTH`` most recent
        # outputs from the forward call.
        if len(output_list) >= GRAPH_WIDTH:
            output_list = output_list[1:]
        # ``pred`` is the last prediction in the first (and only) batch.
        # Shape: <scalar>.
        pred = np.array(predictions[0, -1].data)[0]

        # Create ``out_array`` to print graph as it is populated.
        output_list.append(pred)
        # Shape is the number of iterations we've made, up until we hit
        # ``width`` iterations, after which the shape is ``(width,)``. 
        out_array = np.stack(output_list)
        os.system("clear")
        out_array = np.concatenate([np.array([-1.5]), out_array, np.array([1.5])])
        plot_to_terminal(out_array)
        
        # Grab inputs and outputs for matplotlib plot.
        print("inputs_raw shape:", inputs_raw.shape)
        if torch.__version__[:5] == "0.3.1":
            inputs_raw = inputs_raw.data
        inputs_raw_array = np.array(inputs_raw[0, -1, :])
        print("``inputs_raw_array`` shape:", inputs_raw_array.shape)
        input_actual = inputs_raw_array[...,0]
        all_outputs.append(pred)
        all_inputs.append(input_actual)


    def matplot(graphs_path, data_filename, dfs, ylabels, column_counts):
        """ Do some path handling and call the ``graph()`` function. """
        assert os.path.isdir(graphs_path)
        filename = os.path.basename(data_filename)
        filename_no_ext = filename.split('.')[0]
        save_path = os.path.join(graphs_path, filename_no_ext + ".svg")
        graph(dfs, ylabels, filename_no_ext, column_counts, None, save_path)
        print("Graph saved to:", save_path)
        # Plot with matplotlib.

    # Stack and cast to ``pd.DataFrame``.
    all_in = np.stack(all_inputs)
    all_out = np.stack(all_outputs)
    print("``all_in`` shape:", all_in.shape)
    print("``all_out`` shape:", all_out.shape)
    all_in = np.reshape(all_in, (all_in.shape[0], 1))
    all_out = np.reshape(all_out, (all_out.shape[0], 1))
    print("shape of all in:", all_in.shape) 
    print("shape of all out:", all_out.shape) 
    assert all_out.shape == all_in.shape
    diff = np.concatenate((all_out, all_in), axis=1)
    print("diff shape:", diff.shape)
    df = pd.DataFrame(diff)
    df.columns = ["pred", "actual"]
    print(df)

    dfs = [df]
    y_label = 'Predictions vs Input'
    column_counts = [2]
    # MATPLOT GOES HERE
    matplot(GRAPH_PATH, DATA_FILENAME, dfs, y_label, column_counts)
 
if __name__ == "__main__":
    main()
