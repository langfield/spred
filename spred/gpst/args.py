import argparse

def train_args(parser):
    parser.add_argument(
        "--model_name", type=str, default="openai-gpt", help="pretrained model name"
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model \
        predictions and checkpoints will be written.",
    )
    parser.add_argument("--train_dataset", type=str, default="")

    # Unused args.
    parser.add_argument("--eval_dataset", type=str, default="")
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # ?
    parser.add_argument("--num_train_epochs", type=int, default=3)

    # Optuna args.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=6.25e-5)
    parser.add_argument("--warmup_proportion", type=float, default=0.002)
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--timeout", type=float, default=0)

    # Added.
    parser.add_argument(
        "--gpst_model",
        default=None,
        type=str,
        required=True,
        help="OpenAIGPT pre-trained model path.",
    )
    parser.add_argument(
        "--weights_name",
        default=None,
        type=str,
        required=True,
        help="Name to save weights file.",
    )
    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        required=True,
        help="Name to save config file.",
    )
    parser.add_argument(
        "--data_is_example",
        type=bool,
        default=False,
        help="Whether data is example/sample data or real price data.",
    )
    parser.add_argument(
        "--stationarize",
        dest="stationarize",
        action="store_true",
        help="Whether to stationarize the time series or not.",
    )
    parser.add_argument(
        "--aggregation_size",
        dest="aggregation_size",
        type=int,
        default=1,
        help="How many timesteps to aggregate along series.",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="Whether to normalize input data.",
    )
    parser.set_defaults(stationarize=False)
    parser.add_argument(
        "--save_freq",
        default=1,
        type=int,
        required=False,
        help="Model will be saved after every ``--save_freq`` epochs.",
    )

    return parser
