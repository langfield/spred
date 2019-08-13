import os
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modeling_openai import OpenAIGPTLMHeadModel, OpenAIGPTConfig
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from termplt import plot_to_terminal

if torch.__version__[:5] == "0.3.1":
    from torch.autograd import Variable
    from torch_addons.sampler import SequentialSampler
else:
    from torch.utils.data import SequentialSampler

DEBUG = False


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
        model = load_model()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(device)

    # Set hyperparameters.
    DIM = model.config.n_embd
    MAX_SEQ_LEN = model.config.n_positions
    BATCH_SIZE = 1
    TOTAL_DATA_LEN = None
    PLOT = False
    WIDTH = 100
    print("Data dimensionality:", DIM)
    print("Max sequence length :", MAX_SEQ_LEN)
    print("Eval batch size:", BATCH_SIZE)

    """
    # Get sample data.
    tensor_data = create_sample_data(DIM, MAX_SEQ_LEN, WIDTH, PLOT)
    inputs_raw = tensor_data.contiguous()
    """
    
    # Grab training data.
    raw_data = pd.read_csv("sample_data.csv")
    assert len(raw_data) >= MAX_SEQ_LEN
    output_list = []

    # Iterate in step sizes of 1 over ``raw_data``.
    for i in range(len(raw_data) - MAX_SEQ_LEN):
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
            inputs_raw = inputs_raw.cuda()
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
        
        GRAPH_WIDTH = 50
        if len(output_list) >= GRAPH_WIDTH:
            output_list = output_list[1:]
        pred = np.array(predictions[0, -1, :].data)
        # Average over ``DIM`` (last) dimension.
        pred = np.sum(pred, -1) / DIM
        output_list.append(pred)
        out_array = np.stack(output_list)
        os.system("clear")
        out_array = np.concatenate([np.array([-1.5]), out_array, np.array([1.5])])
        plot_to_terminal(out_array)

if __name__ == "__main__":
    main()
