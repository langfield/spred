import os
import datetime
from typing import Tuple


def get_log(precipitation: str) -> Tuple["TextIOWrapper"]:
    """
    Creates and returns log objects.

    Parameters
    ----------
    precipitation : ``str``.
        Indication of the script from which the log originates.
        e.g. ``rain``, ``snow``, ``eval``.

    Returns
    -------
    log : ``TextIOWrapper``.
        Log object for ``trainer.py``. Write via ``log.write(<str>)``.
    """

    # HARDCODE
    with open("abc/google-10000-english.txt", "r", encoding="utf-8") as english:
        tokens = [word.rstrip() for word in english.readlines()]
    tokens.sort()
    tokens = [word for word in tokens if len(word) > 5]
    if not os.path.isdir("logs/"):
        os.mkdir("logs/")
    dirlist = os.listdir("logs/")
    token_idx = 0
    while 1:
        token = tokens[token_idx]
        already_used = False
        for filename in dirlist:
            if token in filename:
                already_used = True
                break
        if already_used:
            token_idx += 1
            continue
        break
    date = str(datetime.datetime.now())
    date = date.replace(" ", "_")
    log_path = "logs/%s_%s_%s_log.log" % (token, date, precipitation)

    log_dir = os.path.dirname(log_path)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    return log_path
