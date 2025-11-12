import logging
from contextlib import contextmanager
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Process, Queue
from pathlib import Path


try:
    from spack.extensions.mpi.constants import HEAD_NODE_LOGGER_NAME
except ImportError:
    from constants import HEAD_NODE_LOGGER_NAME


def attach_queue_to_logger(queue: Queue):
    logger = logging.getLogger(HEAD_NODE_LOGGER_NAME)
    if not any(isinstance(h, QueueHandler) for h in logger.handlers):
        logger.addHandler(QueueHandler(queue))


class LoggingProcess(Process):
    def __init__(self, log_queue: Queue, **kwargs):
        self.log_queue = log_queue
        super().__init__(**kwargs)
    def run(self):
        attach_queue_to_logger(self.log_queue)
        return super().run()


@contextmanager
def setup_logging_queue(log_name: str, log_file: Path, log_level = logging.DEBUG):
    """
    This sets up a multiprocessing safe logger
    """
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    queue = Queue()
    queue_handler = QueueHandler(queue)
    logger = logging.getLogger(log_name)
    logger.addHandler(queue_handler)
    logger.setLevel(log_level)
    listener = QueueListener(queue, file_handler)
    listener.start()
    try:
        yield queue
    finally:
        listener.stop()
        file_handler.close()
