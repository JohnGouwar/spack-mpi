from multiprocessing.connection import Connection
from multiprocessing import Process, Pipe
from argparse import Namespace
try:
    from spack.extensions.mpi.servers import subprocess_server
    from spack.extensions.mpi.constants import HEAD_RANK_ID
except:
    from servers import subprocess_server
    from constants import HEAD_RANK_ID

def setup_worker_rank(args: Namespace) -> Connection:
    '''
    Sets up the the subprocess server for worker rank
    '''
    parent, child = Pipe()
    Process(
        target=subprocess_server,
        args=(child,)
    ).start()
    return parent

def worker_rank(subprocess_pipe: Connection):
    def handle_task(task):
        pass
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    
    while True:
        task = comm.recv(source=HEAD_RANK_ID)
        if task is None:
            return
        data = handle_task(task)
        comm.send(data, dest=HEAD_RANK_ID)
        
