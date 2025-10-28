from multiprocessing import connection
from subprocess import run, PIPE
from multiprocessing.connection import Connection
from PosixMQ import PosixMQ
from spack.package_base import PackageBase
from spack.installer import PackageInstaller
from dataclasses import dataclass

@dataclass
class CompilerTask:
    mode: str
    output_fifo: str
    cmd: list[str]

def subprocess_server(pipe: Connection):
    '''
    MPI code cannot fork exec directly, forking this before calling MPI_Init
    allows it to work over IPC
    '''
    while True:
        args = pipe.recv()
        if args is None:
            return
        res = run(args, stdout=PIPE, stderr=PIPE)
        pipe.send(res)

def listener_server(
        mq_name: str,
        task_queue_writer: Connection
):
    def _parse_task(task_str: str):
        mode, output_fifo, *cmd = task_str.split(";")
        return CompilerTask(mode, output_fifo, cmd)
    mq = PosixMQ.create(mq_name)
    try:
        while True:
            task_str = mq.recv()
            if task_str == "Done":
                return
            task = _parse_task(task_str)
            task_queue_writer.send(task)
    finally:
        task_queue_writer.close()
        mq.unlink()
        
def installer(
        mq_name: str,
        packages: list[PackageBase]
):
    try:
        PackageInstaller(packages).install()
    except Exception as e:
        raise e
    finally:
        mq = PosixMQ.open(mq_name)
        mq.send("Done", 2)
        mq.close()
