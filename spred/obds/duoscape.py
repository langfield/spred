""" Tests the modified TorRequest class. """
#! /usr/bin/python3
import os
from tor_request import TorRequest

# url = "https://api.cryptowat.ch/markets/kraken/ethusd/orderbook"


def get_ip(tor):
    """ Grab ip address of exit node. """
    url = "http://httpbin.org/ip"
    try:
        response = tor.get(url)
        content = response.text
        return content
    except Exception as exc:
        print(exc)
        raise ValueError(str(exc))


def main():
    """ Testing function. """
    pid = os.fork()
    if pid == 0:
        with TorRequest(proxy_port=9050, ctrl_port=9051, data_dir="./tordatac") as tor:
            for _ in range(10):
                print("child: ", get_ip(tor))  # child
    else:
        with TorRequest(proxy_port=9060, ctrl_port=9061, data_dir="./tordatap") as tor:
            for _ in range(10):
                print("parent: ", get_ip(tor))  # parent

        os.waitpid(pid, 0)

if __name__ == "__main__":
    main()
