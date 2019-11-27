from multiprocessing import Process, Queue, cpu_count
from queue import Full as QueueFull
from queue import Empty as QueueEmpty


def worker(inq, outq):
    for func, args in iter(inq.get, None):
        result = func(*args)
        outq.put(result)


def pool_imap_unordered(
    function: Callable[[Queue, Queue], Any], iterable: Iterable[Any], procs: int
) -> Generator[Any, None, None]:

    # Create queues for sending/receiving items from iterable.
    outq = Queue(procs)
    inq = Queue()

    # Start worker processes.

    for rpt in range(procs):
        Process(target=worker, args=(outq, inq)).start()

    # Iterate iterable and communicate with worker processes.

    inq_len = 0
    outq_len = 0
    itr = iter(iterable)

    try:
        value = next(itr)
        while True:
            try:
                outq.put((function, value), True, 0.1)
                inq_len += 1
                value = next(itr)
            except QueueFull:
                while True:
                    try:
                        result = inq.get(False)
                        outq_len += 1
                        yield result
                    except QueueEmpty:
                        break
    except StopIteration:
        pass

    # Collect all remaining results.

    while outq_len < inq_len:
        result = inq.get()
        outq_len += 1
        yield result

    # Terminate worker processes.

    for rpt in range(procs):
        outq.put(None)
