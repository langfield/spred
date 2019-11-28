from queue import Full as QueueFull
from queue import Empty as QueueEmpty
from typing import Callable, Iterable, Any, Generator
from multiprocessing import Process, Queue
# pylint: disable=bad-continuation


def worker(outq, inq):
    for func, args in iter(outq.get, None):
        result = func(args)
        inq.put(result)


def pool_imap_unordered(
    function: Callable[[Queue, Queue], Any], iterable: Iterable[Any], procs: int
) -> Generator[Any, None, None]:

    # Create queues for sending/receiving items from iterable.
    inq = Queue(procs)
    outq = Queue()

    # Start worker processes.

    for _ in range(procs):
        Process(target=worker, args=(inq, outq)).start()

    # Iterate iterable and communicate with worker processes.

    outq_len = 0
    inq_len = 0
    itr = iter(iterable)

    try:
        value = next(itr)
        while True:
            try:
                inq.put((function, value), True, 0.1)
                outq_len += 1
                value = next(itr)
            except QueueFull:
                while True:
                    try:
                        result = outq.get(False)
                        inq_len += 1
                        yield result
                    except QueueEmpty:
                        break
    except StopIteration:
        pass

    # Collect all remaining results.

    while inq_len < outq_len:
        result = outq.get()
        inq_len += 1
        yield result

    # Terminate worker processes.

    for rpt in range(procs):
        inq.put(None)
