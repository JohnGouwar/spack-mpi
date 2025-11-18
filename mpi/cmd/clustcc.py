from argparse import ArgumentParser
from pathlib import Path

import mpi4py
from spack.cmd.common import arguments

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI  # noqa: E402
import logging
try:
    from spack.extensions.mpi.constants import HEAD_NODE_LOGGER_NAME
    from spack.extensions.mpi.head_rank import HeadRankTaskServer, MpiHeadRank
    from spack.extensions.mpi.logs import setup_logging_queue
    from spack.extensions.mpi.worker_rank import ForkServer, MpiWorkerRank
except ImportError:
    from constants import HEAD_NODE_LOGGER_NAME
    from head_rank import HeadRankTaskServer, MpiHeadRank
    from logs import setup_logging_queue
    from worker_rank import ForkServer, MpiWorkerRank


def setup_parser(parser: ArgumentParser):
    """
    Required method for configuring parser for spack command
    """
    parser.add_argument("--logging-level", choices=["debug", "warning", "error", "none"], default="none")
    subparsers = parser.add_subparsers(dest="subcommand")
    head_parser = subparsers.add_parser("head")
    arguments.add_common_arguments(head_parser, ["specs"])
    head_parser.add_argument("--spec-json", help="Concretized spec to test")
    head_parser.add_argument("--clustcc-spec-json", help="Concretized wrapper spec")
    head_parser.add_argument(
        "--local-concurrent-tasks", help="Concurrent tasks running on head node"
    )
    subparsers.add_parser("worker")


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
    if args.subcommand == "head":
        try:
            with setup_logging_queue(
                    HEAD_NODE_LOGGER_NAME, Path("head_node.log"), logging_level
            ) as logging_queue:
                spec_json_path = Path(args.spec_json) if args.spec_json else None
                clustcc_json_path = (
                    Path(args.clustcc_spec_json) if args.clustcc_spec_json else None
                )
                hrts = HeadRankTaskServer(
                    specs=args.specs,
                    spec_json=spec_json_path,
                    clustcc_spec_json=clustcc_json_path,
                    concurrent_tasks=int(args.local_concurrent_tasks),
                    logging_queue=logging_queue,
                )
                MPI.Init()
                MpiHeadRank(hrts).run()
        except Exception as e:
            raise e
        finally:
            if MPI.Is_initialized():
                MPI.Finalize()
    elif args.subcommand == "worker":
        forkserver = ForkServer()
        try:
            MPI.Init()
            MpiWorkerRank(forkserver, logging_level).run()
        finally:
            MPI.Finalize()
