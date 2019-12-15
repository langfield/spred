#! /usr/bin/python3

import os
import sys
import time
import json
import argparse
import datetime
import functools
import multiprocessing as mp
from time import mktime
from queue import Full as QueueFull
from queue import Empty as QueueEmpty
from typing import List, Any, Dict, Tuple, Generator
from contextlib import closing

from tor_request import TorRequest

# url = "https://api.cryptowat.ch/markets/kraken/ethusd/orderbook"

def get_ip(tor):
    url = "http://httpbin.org/ip"
    try:
        response = tor.get(url)
        content = response.text
        return content
    except Exception as exc:
        print(exc)
        raise ValueError(str(exc))

pid = os.fork()
if pid == 0:
    with TorRequest(proxy_port=9050, ctrl_port=9051, data_dir = './tordatac') as tor:
        for i in range(10):
            print('child: ', get_ip(tor)) # child
else:
    with TorRequest(proxy_port=9060, ctrl_port=9061, data_dir = './tordatap') as tor:
        for i in range(10):
            print('parent: ', get_ip(tor)) # parent
    
    
    os.waitpid(pid, 0)
