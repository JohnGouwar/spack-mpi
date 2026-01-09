from dataclasses import dataclass
import os
from typing import Optional
from epic import PosixShm

try:
    from spack.extensions.mpi.constants import MSG_SEP
except ImportError:
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
    rc: Optional[int]  # return code, none means subprocess didn't even launch
    working_dir: str  # working dir this task was initiated from
    output_fifo: str  # fifo that is masquerading the call
    stdout: Optional[bytes]  # potential stdout of remote process
    stderr: Optional[bytes]  # potential stderr of remote process
    output_bytes: Optional[int]  # number of bytes of the output object file
    output_filename: Optional[str]  # name of output object file
    cmd: Optional[list[str]]  # send back the command to run locally if failed

def _mode_from_cmd(cmd: list[str]) -> str:
    if cmd[0] in ["ld", "ld.gold", "ld.lld"]:
        return "ld"
    if cmd[0] == "cpp":
        return "cpp"
    for arg in cmd:
        if arg == "-E":
            return "cpp"
        elif arg == "-S":
            return "as"
        elif arg == "-c":
            return "cc"
        elif arg in ["-v", "-V", "--version", "-dumpversion"]:
            return "vcheck"
    return "ccld"

def parse_task_from_message(msg: str) -> RawCompilerTask:
    shm_name, size = msg.split(MSG_SEP)
    shm = PosixShm.open(shm_name, int(size))
    data = shm.read().tobytes().decode()
    try:
        wd, fifo, *cmd = data.split(MSG_SEP)
        mode = _mode_from_cmd(cmd)
        return RawCompilerTask(mode, wd, fifo, cmd)
    except Exception as e:
        print(f"Failed to parse {msg} as a compile command")
        raise e
    finally:
        shm.close()
        shm.unlink()


def refine_compiler_task(
    task: RawCompilerTask,
) -> LocalCompilerTask | LocalPreprocessorTask:
    if task.mode == "cc":
        return LocalPreprocessorTask(
            working_dir=task.working_dir,
            output_fifo=task.output_fifo,
            orig_cmd=task.cmd,
        )
    return LocalCompilerTask(task.working_dir, task.output_fifo, task.cmd)
