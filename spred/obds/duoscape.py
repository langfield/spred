""" Tests the modified TorRequest class. """
#!/usr/bin/python3
import os
import shutil
import tempfile
from tor_request import TorRequest


def get_ip(tor: TorRequest) -> str:
    """ Grab ip address of exit node. """
    url = "http://httpbin.org/ip"
    try:
        response = tor.get(url)
        content = response.text
        return content
    except Exception as exc:
        print(exc)
        raise ValueError(str(exc))


def main() -> None:
    """ Testing function. """
    pid = os.fork()
    if pid == 0:
        data_dir = tempfile.mkdtemp()
        with TorRequest(proxy_port=9060, ctrl_port=9061, data_dir=data_dir) as tor:
            for _ in range(10):
                print("child: ", get_ip(tor))  # child
        shutil.rmtree(data_dir)
    else:
        data_dir = tempfile.mkdtemp()
        with TorRequest(proxy_port=9070, ctrl_port=9071, data_dir=data_dir) as tor:
            for _ in range(10):
                print("parent: ", get_ip(tor))  # parent
        shutil.rmtree(data_dir)
        os.waitpid(pid, 0)


if __name__ == "__main__":
    main()
