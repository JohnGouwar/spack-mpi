from argparse import ArgumentParser
from pathlib import Path

import mpi4py
from mpi.logs import setup_logging_queue
from spack.cmd.common import arguments

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI  # noqa: E402

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
    subparsers = parser.add_subparsers(dest="subcommand")
    head_parser = subparsers.add_parser("head")
    arguments.add_common_arguments(head_parser, ["specs"])
    head_parser.add_argument(
        "--mq-name",
        help="Name of the Posix message queue the wrapper will contact",
        default="/spackclustccmq",
    )
    head_parser.add_argument("--spec-json", help="Concretized spec to test")
    head_parser.add_argument("--clustcc-spec-json", help="Concretized wrapper spec")
    head_parser.add_argument(
        "--local-concurrent-tasks", help="Concurrent tasks running on head node"
    )
    subparsers.add_parser("worker")


def clustcc(parser, args):
    if args.subcommand == "head":
        try:
            with setup_logging_queue(HEAD_NODE_LOGGER_NAME, Path("head_node.log")) as logging_queue:
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
            MpiWorkerRank(forkserver).run()
        finally:
            MPI.Finalize()
