import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def setup_logging(log_file: Path):
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    log_queue = Queue()
    queue_handler = QueueHandler(log_queue)
    root = logging.getLogger()
    root.addHandler(queue_handler)
    root.setLevel("NOTSET")
    listener = QueueListener(log_queue, file_handler) 
    listener.start()
    try:
        yield
    finally:
        listener.stop()
    
    
