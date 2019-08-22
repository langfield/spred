""" Get arguments for ``train.py`` and ``eval.py``. """
import argparse


def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ Adds GPST arguments to the passed Parser object. """

    # Required.
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpst_model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    # Saving.
    parser.add_argument("--model_name", type=str, default="openai-gpt")
    parser.add_argument("--save_freq", default=1, type=int)

    # Evaluation.
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--graph_dir", type=str, default="graphs/")
    parser.add_argument("--terminal_plot_width", type=int, default=50)

    # Training.
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=6.25e-5)
    parser.add_argument("--warmup_proportion", type=float, default=0.002)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--timeout", type=float, default=0)

    # Data preprocessing.
    parser.add_argument("--aggregation_size", type=int, default=1)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--stationarize", action="store_true")
    parser.set_defaults(stationarize=False)

    return parser


DOCSTRING = """
    Arguments
    ---------
    model_name : str, optional (default = "openai-gpt").
        The name used to save weights and config. These files will be saved
        in ``<args.output_dir>/<args.model_name>.bin``, and
        ``<args.output_dir>/<args.model_name>.json``, respectively.
    output_dir : str, required.
        "The output directory where the model predictions and checkpoints will be written.
    dataset : str, required.
        The path to the dataset used for training/evaluation.
    width : int, optional (default = 100).
        Number of timesteps to print during evaluation.
    eval_batch_size : int, optional (default = 1).
        Batch size used for evaluation.
    graph_dir : str, optional (default = "graphs/").
        Path to save graphs to during evaluation.
    terminal_plot_width : int, optional (default = 50).
        How many timesteps to fit along the x-axis in ``terminal_plot``
        during evaluation.
    num_train_epochs : int, optional (default = 3).
        Number of epochs (complete passes through the entire dataset) to run
        during training.
    seed : int, optional (default = 42).
        Random seed to use for torch and numpy.
    train_batch_size : int, optional.
        Batch size (measured in sequences) used during training.
    max_grad_norm : int, optional (default = 1).
        Max norm of gradients during training.
    learning_rate : float, optional.
        Maximum/peak learning rate during scheduling.
    warmup_proportion : float, optional.
        Proportion of total training time during which learning rate is
        changed from 0 to ``args.learning_rate`` linearly.
    warmup_steps : int, optional.
        Number of sequences to warmup over. Can be used instead instead of
        ``args.warmup_proportion``.
    lr_schedule : str, optional.
        Which learning rate annealing schedule to use.
    weight_decay : float, optional.
        Rate of weight decay during training.
    adam_epsilon : float, optional.
        Epsilon value to use for Adam optimizer.
    timeout : int, optional.
        Number of seconds after which the training is automatically stopped.
        Setting this argument to ``0`` allows the training to run for as many
        epochs as is specified, regardless of the running time.
    gpst_model : str, required.
        Path to the initial ``pytorch_transformers``-style config file used to
        spin up the model, or the path to a trained model file.
    stationarize : flag.
        Whether to stationarize the time series or not.
    aggregation_size : int, optional.
        How many timesteps to aggregate along first dimension of raw data.
    normalize : flag.
        Whether to normalize the raw data.
    save_freq : int, optional.
        Model will be saved after every ``args.save_freq`` epochs.
"""
