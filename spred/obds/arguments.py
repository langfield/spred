""" Get arguments for ``format.py``. """
import argparse


def strbool(value: str) -> bool:
    """
    Converts a string into a boolean value.

    Parameters
    ----------
    value : ``int``.
        The string to check if boolean.

    Returns
    -------
    <bool> : ``bool``.
        True or False based on the string parse.

    Raises
    ------
    ArgumentTypeError
        If the parse fails, i.e. the input is invalid.
    """

    # pylint: disable=no-else-return
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds arguments for running ``format.py`` via ``gale.sh`` to the passed
    ``ArgumentParser`` object.

    Parameters
    ----------
    parser : ``argparse.ArgumentParser``, required.
        A parser object, possibly with existing arguments already added.

    Returns
    -------
    parser : ``argparse.ArgumentParser``.
        The same parser as was passed in as a parameter, but with all
        the arguments below added.
    """

    # Print statistics.
    parser.add_argument("--print_stats", type=strbool, required=True)
    parser.add_argument("--hour", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--bound", type=float, required=True)
    parser.add_argument("--analysis_depth", type=int, required=True)
    parser.add_argument("--num_gap_freq_levels", type=int, required=True)
    parser.add_argument("--num_gap_freq_sizes", type=int, required=True)
    parser.add_argument("--fig_dir", type=str, required=True)

    # Compute k-value.
    parser.add_argument("--compute_k", type=strbool, required=True)
    parser.add_argument("--hours", type=int, required=True)
    parser.add_argument("--sigma", type=int, required=True)

    return parser
