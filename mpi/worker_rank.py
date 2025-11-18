import logging
import os
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path
from subprocess import PIPE, CompletedProcess, run
from typing import Optional
from tempfile import TemporaryDirectory

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI  # noqa: E402

try:
    from spack.extensions.mpi.constants import HEAD_RANK_ID, WorkerResponseTag
    from spack.extensions.mpi.task import RemoteCompilerResponse, RemoteCompilerTask
except ImportError:
    from constants import HEAD_RANK_ID, WorkerResponseTag
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

    def shutdown(self):
        self.parent_pipe.send(None)

    def spawn(self, args: list[str], **kwargs) -> CompletedProcess:
        self.parent_pipe.send((args, kwargs))
        return self.parent_pipe.recv()


def normalize_cmd_args(args: list[str], td: str) -> tuple[str, str, str, list[str]]:
    input_filename = None
    output_filename = None
    original_output_filename = None
    normalized_args = list(args)
    for i, arg in enumerate(args):
        if arg.endswith(".c") or arg.endswith(".cc") or arg.endswith(".cpp"):
            input_filename = Path(arg).name
            normalized_args[i] = os.path.join(td, input_filename)
        elif arg == "-o":
            original_output_filename = args[i + 1]
            output_filename = str(Path(td) / Path(original_output_filename).name)
            normalized_args[i + 1] = output_filename
    if output_filename is None:
        original_output_filename = "a.out"
        output_filename = "a.out"
    if input_filename:
        assert output_filename is not None
        assert original_output_filename is not None
        return (
            input_filename,
            original_output_filename,
            output_filename,
            normalized_args,
        )
    else:
        raise ValueError("Failed to normalize remote command args")


class MpiWorkerRank:
    def __init__(self, forkserver: ForkServer, logging_level):
        assert MPI.Is_initialized()
        self.world_comm = MPI.COMM_WORLD.Dup()
        self.fork_server = forkserver
        logging.basicConfig(
            filename=f"worker_{self.world_comm.Get_rank()}.log",
            filemode="w",
            format="%(asctime)s - %(message)s",
            level=logging_level
        )

    def handle_cc_args(
            self, task: RemoteCompilerTask, td: str
    ) -> tuple[RemoteCompilerResponse, Optional[bytes]]:
        try:
            (infile, orig_outfile, outfile, norm_args) = normalize_cmd_args(task.orig_cmd, td)
            with open(infile, "w") as f:
                f.write(task.input_file_text)
            res = self.fork_server.spawn(norm_args, stderr=PIPE, stdout=PIPE)
            if res.returncode == 0:
                logging.debug(f"Running {norm_args} succeeded")
                with open(outfile, "rb") as f:
                    output_bytes = f.read()
            else:
                logging.debug(f"Running {norm_args} failed, {res}")
                output_bytes = None
            return RemoteCompilerResponse(
                rc=res.returncode,
                output_fifo=task.output_fifo,
                working_dir=task.working_dir,
                stdout=res.stdout if len(res.stdout) > 0 else None,
                stderr=res.stderr if len(res.stdout) > 0 else None,
                output_bytes=len(output_bytes) if output_bytes else None,
                output_filename=orig_outfile,
                cmd=None if res.returncode == 0 else task.orig_cmd,
            ), output_bytes
        except Exception as e:
            logging.debug(f"handle_cc_args for {task.orig_cmd} failed with {e}")
            return RemoteCompilerResponse(
                rc=None,
                output_fifo=task.output_fifo,
                working_dir=task.working_dir,
                stdout=None,
                stderr=None,
                output_bytes=None,
                output_filename=None,
                cmd=task.orig_cmd,
            ), None

    def run(self):
        send_reqs = [MPI.REQUEST_NULL, MPI.REQUEST_NULL]
        with TemporaryDirectory(dir="/tmp") as td:
            os.chdir(td)
            while True:
                remote_cc_task: Optional[RemoteCompilerTask] = self.world_comm.recv(
                    source=HEAD_RANK_ID
                )
                if remote_cc_task is None:
                    logging.debug("Received sentinel None, shutting down...")
                    self.fork_server.shutdown()
                    return
                logging.debug(f"Received: {remote_cc_task.orig_cmd}")
                (resp, object_bytes) = self.handle_cc_args(remote_cc_task, td)
                MPI.Request.Waitall(send_reqs)
                send_reqs[0] = self.world_comm.isend(
                    resp, dest=HEAD_RANK_ID, tag=WorkerResponseTag.RESPONSE
                )
                logging.debug(f"Sending response: {resp}")
                if object_bytes:
                    logging.debug(f"Sending {len(object_bytes)} bytes")
                    send_reqs[1] = self.world_comm.Isend(
                        [object_bytes, MPI.BYTE],
                        dest=HEAD_RANK_ID,
                        tag=WorkerResponseTag.OBJECT_BYTES,
                    )
                else:
                    send_reqs[1] = MPI.REQUEST_NULL
