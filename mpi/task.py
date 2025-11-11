from dataclasses import dataclass
from typing import Optional

try:
    from spack.extensions.mpi.constants import MSG_SEP
except:
    from constants import MSG_SEP


@dataclass
class RawCompilerTask:
    mode: str  # vcheck, cc, cpp, ld, and ccld
    working_dir: str
    output_fifo: str
    cmd: list[str]


@dataclass
class LocalCompilerTask:
    working_dir: str
    output_fifo: str
    cmd: list[str]


@dataclass
class LocalPreprocessorTask:
    working_dir: str
    output_fifo: str
    orig_cmd: list[str]


@dataclass
class RemoteCompilerTask:
    input_file_text: str  # include processed input file
    working_dir: str  # working dir this was called from (used in reply)
    output_fifo: str  # where to eventually write the output on host (used in reply)
    orig_cmd: list[str]  # the raw command to execute


@dataclass
class RemoteCompilerResponse:
    rc: int  # return code
    working_dir: str  # working dir this task was initiated from
    output_fifo: str  # fifo that is masquerading the call
    stdout: Optional[bytes]  # potential stdout of remote process
    stderr: Optional[bytes]  # potential stderr of remote process
    output_bytes: Optional[bytes]  # bytes of the output object file
    output_filename: Optional[str]  # name of output object file
    cmd: Optional[list[str]]  # send back the command to run locally if failed


def parse_task_from_message(msg: str) -> RawCompilerTask:
    mode, wd, output_fifo, *cmd = msg.split(MSG_SEP)
    return RawCompilerTask(mode, wd, output_fifo, cmd)


def refine_compiler_task(
    task: RawCompilerTask,
) -> LocalCompilerTask | LocalPreprocessorTask:
    return LocalCompilerTask(task.working_dir, task.output_fifo, task.cmd)
