from multiprocessing.connection import Connection
from multiprocessing import Process, Pipe
from argparse import Namespace
from spack.cmd import parse_specs
try:
    from spack.extensions.mpi.servers import subprocess_server, listener_server, installer
    from spack.extensions.mpi.swap import concretize_with_clustcc
    from spack.extensions.mpi.constants import HEAD_RANK_ID
except:
    from servers import subprocess_server, listener_server, installer
    from swap import concretize_with_clustcc
    from constants import HEAD_RANK_ID

def setup_head_rank(args: Namespace) -> tuple[Connection, Connection]:
    '''
    Pre-MPI setup for head rank. Spawns:
    1. PosixMQ listener for tasks from clustcc-client
    2. A subprocess server for running external commands
    3. The spack installer
    '''
    # Listener proc
    task_queue_read, task_queue_write = Pipe(duplex=False)
    Process(
        target=listener_server,
        args=(args.mq_name, task_queue_write,)
    ).start()
    task_queue_write.close()
    # Subprocess server
    parent_subprocess_pipe, child_subprocess_pipe = Pipe()
    Process(
        target=subprocess_server,
        args=(child_subprocess_pipe,)
    ).start()
    # Installer proc
    specs = parse_specs(args.specs)
    clustcc_specs = concretize_with_clustcc(specs)
    packages = [c.package for c in clustcc_specs]
    Process(
        target=installer,
        args=(args.mq_name, packages)
    ).start()
    return (task_queue_read, parent_subprocess_pipe)

def head_rank(
        task_queue: Connection,
        subprocess_pipe: Connection
):
    '''
    task_queue: Tasks from the clustcc-client listener
    subprocess_pipe: Pipe to write subprocess executions
    '''
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    assert comm.Get_rank() == HEAD_RANK_ID
    outstanding = [False] * (comm.Get_size() - 1)
    # task_queue.recv() will fail with EOFError once its write end has been closed
    try: 
        # Spawn initial tasks
        for idx, worker_rank in enumerate(range(1, comm.Get_size())):
                t = task_queue.recv()
                comm.send(t, dest=worker_rank)
                outstanding[idx] = True
        # Round robbin listen for completed tasks, ensures fairness
        while True:
            for idx, worker_rank in enumerate(range(1, comm.Get_size())):
                if outstanding[idx]:
                    flag = comm.iprobe(source=worker_rank)
                    if flag:
                        data = comm.recv(source=worker_rank)
                        if task_queue.poll(0):
                            t = task_queue.recv()
                            comm.send(t, dest=worker_rank)
                        else:
                            outstanding[idx] = False
    except EOFError:
        # When we run out of tasks, make sure to complete the ones that are still outstanding
        for idx, worker_rank in enumerate(range(1, comm.Get_size())):
            if outstanding[idx]:
                data = comm.recv(source=worker_rank)
                comm.send(None, dest=worker_rank)
        return
