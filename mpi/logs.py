import logging
import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def setup_logging(log_name: str, log_file: Path, log_level: str = "NOTSET"):
    """
    This sets up a multiprocessing safe logger
    """
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    log_queue = Queue()
    queue_handler = QueueHandler(log_queue)
    logger = logging.getLogger(log_name)
    logger.addHandler(queue_handler)
    logger.setLevel(log_level)
    listener = QueueListener(log_queue, file_handler)
    listener.start()
    try:
        yield
    finally:
        listener.stop()
