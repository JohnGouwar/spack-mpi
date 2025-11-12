from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path
from subprocess import PIPE, CompletedProcess, run
from typing import Optional

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI  # noqa: E402

try:
    from spack.extensions.mpi.constants import HEAD_RANK_ID
    from spack.extensions.mpi.task import RemoteCompilerResponse, RemoteCompilerTask
except ImportError:
    from constants import HEAD_RANK_ID
    from task import RemoteCompilerResponse, RemoteCompilerTask


class ForkServer:
    """
    MPI code cannot fork exec directly, forking this before calling MPI_Init
    allows it to work over IPC
    """

    @staticmethod
    def subprocess_server(pipe: Connection):
        while True:
            msg = pipe.recv()
            if msg is None:
                return
            args, kwargs = msg
            res = run(args, **kwargs)
            pipe.send(res)

    def __init__(self):
        assert not MPI.Is_initialized(), "ForkServer must be started before MPI.Init()"
        parent_pipe, child_pipe = Pipe()
        self.parent_pipe = parent_pipe
        self.server = Process(target=ForkServer.subprocess_server, args=(child_pipe,))
        self.server.start()

    def spawn(self, args: list[str], **kwargs) -> CompletedProcess:
        self.parent_pipe.send((args, kwargs))
        return self.parent_pipe.recv()


def normalize_cmd_args(args: list[str]) -> tuple[str, str, str, list[str]]:
    input_filename = None
    output_filename = None
    original_output_filename = None
    normalized_args = list(args)
    for i, arg in enumerate(args):
        if arg.endswith(".c") or arg.endswith(".cc") or arg.endswith(".cpp"):
            input_filename = Path(arg).name
            normalized_args[i] = input_filename
        elif arg == "-o":
            original_output_filename = args[i + 1]
            output_filename = Path(original_output_filename).name
            normalized_args[i + 1] = output_filename
    if input_filename and output_filename and original_output_filename:
        return (
            input_filename,
            original_output_filename,
            output_filename,
            normalized_args,
        )
    else:
        raise ValueError("Failed to normalize remote command args")


class MpiWorkerRank:
    def __init__(self, forkserver: ForkServer):
        assert MPI.Is_initialized()
        self.fork_server = forkserver

    def handle_cc_args(self, task: RemoteCompilerTask) -> RemoteCompilerResponse:
        (infile, orig_outfile, outfile, norm_args) = normalize_cmd_args(task.orig_cmd)
        with open(infile, "w") as f:
            f.write(task.input_file_text)
        res = self.fork_server.spawn(norm_args, stdin=PIPE, stdout=PIPE)
        if res.returncode == 0:
            with open(outfile, "rb") as f:
                output_bytes = f.read()
        else:
            output_bytes = None
        resp = RemoteCompilerResponse(
            rc=res.returncode,
            output_fifo=task.output_fifo,
            working_dir=task.working_dir,
            stdout=res.stdout if len(res.stdout) > 0 else None,
            stderr=res.stderr if len(res.stdout) > 0 else None,
            output_bytes=output_bytes,
            output_filename=orig_outfile,
            cmd=None if res.returncode == 0 else task.orig_cmd,
        )
        return resp

    def run(self):
        world_comm = MPI.COMM_WORLD.Dup()
        while True:
            remote_cc_task: Optional[RemoteCompilerTask] = world_comm.recv(
                source=HEAD_RANK_ID
            )
            if remote_cc_task is None:
                return
            resp = self.handle_cc_args(remote_cc_task)
            world_comm.send(resp, dest=HEAD_RANK_ID)
