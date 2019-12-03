from multiprocessing import Process, Queue, cpu_count
from queue import Full as QueueFull
from queue import Empty as QueueEmpty


def worker(recvq, sendq):
    for func, args in iter(recvq.get, None):
        print("Args:", args)
        result = func(*args)
        sendq.put(result)


def pool_imap_unordered(function, iterable, procs=cpu_count()):
    # Create queues for sending/receiving items from iterable.

    sendq = Queue(procs)
    recvq = Queue()

    # Start worker processes.

    for rpt in range(procs):
        Process(target=worker, args=(sendq, recvq)).start()

    # Iterate iterable and communicate with worker processes.

    send_len = 0
    recv_len = 0
    itr = iter(iterable)

    try:
        value = next(itr)
        print("Value:", value)
        while True:
            try:
                sendq.put((function, value), True, 0.1)
                send_len += 1
                value = next(itr)
            except QueueFull:
                while True:
                    try:
                        result = recvq.get(False)
                        recv_len += 1
                        yield result
                    except QueueEmpty:
                        break
    except StopIteration:
        pass

    # Collect all remaining results.

    while recv_len < send_len:
        result = recvq.get()
        recv_len += 1
        yield result

    # Terminate worker processes.

    for rpt in range(procs):
        sendq.put(None)
