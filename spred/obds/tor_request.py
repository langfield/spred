""" Makes requests through the Tor network. """
from __future__ import annotations
import time
import subprocess
from typing import Optional
import stem
import requests
from stem.control import Controller
from stem.process import launch_tor_with_config

# pylint: disable=bad-continuation, missing-function-docstring


class TorRequest:
    """ Opens a Tor process given ports and a data directory. """

    def __init__(
        self,
        proxy_port: int = 9060,
        ctrl_port: int = 9061,
        data_dir: str = "./tor_temp_dir",
        password: Optional[str] = None,
    ) -> None:

        self.proxy_port = proxy_port
        self.ctrl_port = ctrl_port
        self.data_dir = data_dir

        self._tor_proc = None
        if not self._tor_process_exists():
            try:
                self._tor_proc = self._launch_tor()
            except OSError as error:
                print("\n\n!!!ERROR!!!: Try changing the ports +/-10.")
                raise error

        self.ctrl = Controller.from_port(port=self.ctrl_port)
        self.ctrl.authenticate(password=password)

        self.session = requests.Session()
        self.session.proxies.update(
            {
                "http": "socks5://localhost:%d" % self.proxy_port,
                "https": "socks5://localhost:%d" % self.proxy_port,
            }
        )

    # pylint: disable=bare-except
    def _tor_process_exists(self) -> bool:
        try:
            ctrl = Controller.from_port(port=self.ctrl_port)
            ctrl.close()
            return True
        except:
            return False

    def _launch_tor(self) -> subprocess.Popen:
        process: subprocess.Popen = launch_tor_with_config(
            config={
                "SocksPort": str(self.proxy_port),
                "ControlPort": str(self.ctrl_port),
                "DataDirectory": self.data_dir,
            },
            take_ownership=True,
        )
        return process

    # pylint: disable=bare-except
    def close(self) -> None:
        try:
            self.session.close()
        except:
            pass

        try:
            self.ctrl.close()
        except:
            pass

        if self._tor_proc:
            self._tor_proc.terminate()

    # pylint: disable=no-member
    def reset_identity_async(self) -> None:
        self.ctrl.signal(stem.Signal.NEWNYM)

    def reset_identity(self) -> None:
        self.reset_identity_async()
        time.sleep(self.ctrl.get_newnym_wait())

    def get(self, *args: str, **kwargs: str) -> requests.models.Response:
        return self.session.get(*args, **kwargs)

    def post(self, *args: str, **kwargs: str) -> requests.models.Response:
        return self.session.post(*args, **kwargs)

    def put(self, *args: str, **kwargs: str) -> requests.models.Response:
        return self.session.put(*args, **kwargs)

    def patch(self, *args: str, **kwargs: str) -> requests.models.Response:
        return self.session.patch(*args, **kwargs)

    def delete(self, *args: str, **kwargs: str) -> requests.models.Response:
        return self.session.delete(*args, **kwargs)

    def __enter__(self) -> TorRequest:
        return self

    def __exit__(self, *args: str) -> None:
        self.close()
