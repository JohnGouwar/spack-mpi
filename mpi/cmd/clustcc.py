import os
from argparse import ArgumentParser
from pathlib import Path
import subprocess
from epic import PosixMQ, PosixShm

from mpi.constants import MQ_NAME
from mpi.task import LocalCompilerTask, parse_task_from_message
import mpi4py
from spack.cmd.common import arguments

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI  # noqa: E402
import logging
try:
    from spack.extensions.mpi.constants import HEAD_NODE_LOGGER_NAME, DEFAULT_PORTFILE, MQ_NAME, MQ_DONE
    from spack.extensions.mpi.head_rank import HeadRankTaskServer, MpiHeadRank, run_local_compiler_task
    from spack.extensions.mpi.logs import setup_logging_queue
    from spack.extensions.mpi.worker_rank import ForkServer, MpiWorkerRank
    from spack.extensions.mpi.config import parse_config_file, gen_launch_files, gen_empty_config_file
    from spack.extensions.mpi.task import LocalCompilerTask, parse_task_from_message
except ImportError:
    from constants import HEAD_NODE_LOGGER_NAME, DEFAULT_PORTFILE, MQ_NAME, MQ_DONE
    from head_rank import HeadRankTaskServer, MpiHeadRank, run_local_compiler_task
    from logs import setup_logging_queue
    from worker_rank import ForkServer, MpiWorkerRank
    from config import parse_config_file, gen_launch_files, gen_empty_config_file
    from task import LocalCompilerTask, parse_task_from_message

level = "long"
description = "distributed builds on a cluster"
section = "build packages"

def setup_parser(parser: ArgumentParser):
    """
    Required method for configuring parser for spack command
    """
    parser.add_argument(
        "--logging-level",
        choices=["debug", "warning", "error", "none"],
        default="none",
        help="Level for debugging"
    )
    parser.add_argument(
        "--logging-prefix",
        type=Path,
        default=Path("logs"),
        help="Directory to store logs"
    )
    parser.add_argument(
        "--port-file",
        type=Path,
        default=DEFAULT_PORTFILE,
        help="File where port name will be published to link processes"
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    config_parser = subparsers.add_parser("config")
    config_parser.add_argument(
        "config_mode",
        choices=["new", "scripts"],
        help="Generate new config or generate scripts from existing config"
    )
    config_parser.add_argument(
        "config_file",
        help="Path to config json file, created if it does not exist",
        type=Path
    )
    config_parser.add_argument(
        "--is-slurm",
        action="store_true",
        help="Generate a slurm batch file rather than just a call to launch file"
    )
    head_parser = subparsers.add_parser("head")
    arguments.add_common_arguments(head_parser, ["specs", "jobs"])
    head_parser.add_argument("--spec-json", help="Concretized spec to test")
    head_parser.add_argument(
        "--local-concurrent-tasks", help="Concurrent tasks running on head node"
    )
    subparsers.add_parser("worker")
    subparsers.add_parser("test-server", help="Simple local server to help debug clients")


def clustcc(parser, args):
    if args.logging_level == "none":
        logging_level = logging.NOTSET
    elif args.logging_level == "debug":
        logging_level = logging.DEBUG
    elif args.logging_level == "error":
        logging_level = logging.ERROR
    elif args.logging_level == "warning":
        logging_level = logging.WARNING
    else:
        raise Exception(f"Unrecognized logging level: {args.logging_level}")
    args.logging_prefix.mkdir(exist_ok=True, parents=True)
    if args.subcommand == "head":
        try:
            with setup_logging_queue(
                    HEAD_NODE_LOGGER_NAME, args.logging_prefix / "head_node.log", logging_level
            ) as logging_queue:
                spec_json_path = Path(args.spec_json) if args.spec_json else None
                hrts = HeadRankTaskServer(
                    specs=args.specs,
                    spec_json=spec_json_path,
                    concurrent_tasks=int(args.local_concurrent_tasks),
                    logging_queue=logging_queue,
                )
                MPI.Init()
                MpiHeadRank(hrts, port_file=args.port_file).run()
        except Exception as e:
            raise e
        finally:
            if MPI.Is_initialized():
                MPI.Finalize()
    elif args.subcommand == "worker":
        forkserver = ForkServer()
        try:
            MPI.Init()
            MpiWorkerRank(
                forkserver,
                logging_level,
                args.logging_prefix,
                port_file=args.port_file
            ).run()
        finally:
            MPI.Finalize()
    elif args.subcommand == "config":
        if args.config_mode == "new":
            gen_empty_config_file(args.config_file)
            if args.config_file:
                print(f"An empty config file has been generated at: {args.config_file.absolute()}")
        else:
            gen_launch_files(
                parse_config_file(args.config_file),
                Path(os.getcwd()),
                is_slurm=args.is_slurm
            )
    elif args.subcommand == "test-server":
        mq = PosixMQ.create(MQ_NAME)
        try:
            while True:
                msg = mq.recv()
                if msg == MQ_DONE:
                    return
                raw_task = parse_task_from_message(msg)
                local_task = LocalCompilerTask(raw_task.working_dir, raw_task.output_fifo, raw_task.cmd)
                print(local_task)
                run_local_compiler_task(local_task)
        finally:
            mq.close()
            mq.unlink()
                
