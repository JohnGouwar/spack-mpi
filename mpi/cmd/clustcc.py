import os
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Event
from epic import PosixMQ

import mpi4py
import spack.config
from spack.cmd import parse_specs
from spack.cmd.common import arguments
from spack.spec import EMPTY_SPEC

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI  # noqa: E402
import logging

try:
    from spack.extensions.mpi.constants import (
        HEAD_NODE_LOGGER_NAME,
        DEFAULT_PORTFILE,
        MQ_NAME,
        MQ_DONE,
    )
    from spack.extensions.mpi.head_rank import (
        run_local_compiler_task,
        start_head_rank,
    )
    from spack.extensions.mpi.logs import setup_logging_queue, LoggingProcess
    from spack.extensions.mpi.worker_rank import ForkServer, MpiWorkerRank
    from spack.extensions.mpi.config import (
        parse_config_file,
        gen_launch_files,
        gen_empty_config_file,
    )
    from spack.extensions.mpi.task import LocalCompilerTask, parse_task_from_message
    from spack.extensions.mpi.concretize import concretize_with_clustcc
    from spack.extensions.mpi.jsonl import read_specs_from_jsonl, write_specs_to_jsonl
except ImportError as e:
    print(e)
    from constants import HEAD_NODE_LOGGER_NAME, DEFAULT_PORTFILE, MQ_NAME, MQ_DONE
    from head_rank import run_local_compiler_task, start_head_rank
    from ..logs import setup_logging_queue, LoggingProcess
    from worker_rank import ForkServer, MpiWorkerRank
    from config import parse_config_file, gen_launch_files, gen_empty_config_file
    from task import LocalCompilerTask, parse_task_from_message
    from concretize import concretize_with_clustcc
    from jsonl import read_specs_from_jsonl, write_specs_to_jsonl

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
        help="Level for debugging",
    )
    parser.add_argument(
        "--logging-prefix",
        type=Path,
        default=Path("logs"),
        help="Directory to store logs",
    )
    parser.add_argument(
        "--port-file",
        type=Path,
        default=DEFAULT_PORTFILE,
        help="File where port name will be published to link processes",
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    config_parser = subparsers.add_parser("config")
    config_parser.add_argument(
        "config_mode",
        choices=["new", "scripts"],
        help="Generate new config or generate scripts from existing config",
    )
    config_parser.add_argument(
        "config_file",
        help="Path to config json file, created if it does not exist",
        type=Path,
    )
    config_parser.add_argument(
        "--is-slurm",
        action="store_true",
        help="Generate a slurm batch file rather than just a call to launch file",
    )
    head_parser = subparsers.add_parser("head")
    arguments.add_common_arguments(head_parser, ["specs", "jobs"])
    head_parser.add_argument("--spec-jsonl", type=Path, help="Concretized spec to test")
    head_parser.add_argument(
        "--local-concurrent-tasks", help="Concurrent tasks running on head node"
    )
    subparsers.add_parser("worker")
    subparsers.add_parser(
        "test-server", help="Simple local server to help debug clients"
    )


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
                HEAD_NODE_LOGGER_NAME,
                args.logging_prefix / "head_node.log",
                logging_level,
            ) as logging_queue:
                logger = logging.getLogger(HEAD_NODE_LOGGER_NAME)
                installer_start_event = Event()
                LoggingProcess(
                    target=start_head_rank,
                    args=(args.local_concurrent_tasks, logging_queue, args.port_file, installer_start_event),
                    log_queue=logging_queue
                ).start()
                if args.spec_jsonl is not None and args.spec_jsonl.exists():
                    packages = []
                    spec_pairs = read_specs_from_jsonl(args.spec_jsonl)
                    for (_, concrete) in spec_pairs:
                        print(concrete, flush=True)
                        if concrete != EMPTY_SPEC:
                            packages.append(concrete.package)
                else:
                    assert len(args.specs) > 0, "Must build at least one spec"
                    logger.info(f'Concretizing specs passed on the command line')
                    abstract_specs = parse_specs(args.specs)
                    concretized_specs = concretize_with_clustcc(abstract_specs)
                    if args.spec_jsonl is not None:
                        write_specs_to_jsonl(abstract_specs, concretized_specs, args.spec_jsonl)
                    logger.info(f'Concretization finished')
                    packages = [c.package for c in concretized_specs]
                if spack.config.get("config:installer", "old") == "new":
                    from spack.new_installer import PackageInstaller
                else:
                    from spack.installer import PackageInstaller

                installer_start_event.wait()
                logger.info('Starting installer')
                try:
                    PackageInstaller(packages).install()
                except Exception as e:
                    logger.debug(f"Got exception {e} in installer")
                finally:
                    logger.debug(f"Installer sending {MQ_DONE} to listener")
                    mq = PosixMQ.open(MQ_NAME)
                    mq.send(MQ_DONE, 2)
                    mq.close()

        except Exception as e:
            raise e
    elif args.subcommand == "worker":
        forkserver = ForkServer()
        try:
            MPI.Init()
            MpiWorkerRank(
                forkserver, logging_level, args.logging_prefix, port_file=args.port_file
            ).run()
        finally:
            MPI.Finalize()
    elif args.subcommand == "config":
        if args.config_mode == "new":
            gen_empty_config_file(args.config_file)
            if args.config_file:
                print(
                    f"An empty config file has been generated at: {args.config_file.absolute()}"
                )
        else:
            gen_launch_files(
                parse_config_file(args.config_file),
                Path(os.getcwd()),
                is_slurm=args.is_slurm,
            )
    elif args.subcommand == "test-server":
        mq = PosixMQ.create(MQ_NAME)
        try:
            while True:
                msg = mq.recv()
                if msg == MQ_DONE:
                    return
                raw_task = parse_task_from_message(msg)
                local_task = LocalCompilerTask(
                    raw_task.working_dir, raw_task.output_fifo, raw_task.cmd
                )
                print(local_task)
                run_local_compiler_task(local_task)
        finally:
            mq.close()
            mq.unlink()
